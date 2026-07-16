//
//  MetaWearableModule.swift
//  ProgramATApp
//
//  Created by jxr on 5/28/26.
//
//  Native bridge for the Meta Ray-Ban (DAT) camera. Exposes one-time device
//  registration plus a continuous camera pipeline the Tools screen uses when
//  the user selects "Meta Ray-Ban" as their camera source.
//

import Foundation
import React
import UIKit
import MWDATCore
import MWDATCamera

@objc(MetaWearablesModule)
class MetaWearablesModule: NSObject {

    private var registrationTask: Task<Void, Never>?

    // Ray-Ban shared-pipeline streaming. This powers the Tools camera when the
    // user selects "Meta Ray-Ban" as their camera source. Each decoded frame is
    // stored as the latest frame so JS can poll it and feed it through the same
    // WebSocket frame pipeline the phone camera uses.
    private var rayBanSession: DeviceSession?
    private var rayBanStream: MWDATCamera.Stream?
    private var rayBanSessionStateToken: Any?
    private var rayBanSessionErrorToken: Any?
    private var rayBanStreamStateToken: Any?
    private var rayBanStreamErrorToken: Any?
    private var rayBanFrameToken: Any?
    private var latestRayBanImage: UIImage?
    // Diagnostics surfaced via captureRayBanFrame's rejection message so they
    // are visible in Metro without needing the native Xcode console.
    private var rayBanFramesReceived = 0
    private var rayBanFramesDecodeFailed = 0
    private var rayBanLastStreamState = "none"
    private var rayBanLastSessionState = "none"
    private var rayBanLastSessionError: String?
    private var rayBanLastStreamError: String?
    // True from the moment a Ray-Ban stop is requested until the DeviceSession
    // reaches STOPPED and resources are released. Guards against creating a new
    // session before the old one finishes tearing down (sessionAlreadyExists).
    private var rayBanStopping = false

    @objc
    static func requiresMainQueueSetup() -> Bool {
        return false
    }

    // MARK: - Registration

    /// One-time device registration. Launches the Meta AI authorization flow so
    /// the paired Ray-Ban glasses become usable by the app. Registration state
    /// is persisted by the Meta SDK, so this normally only needs to run once.
    @objc
    func registerDevice() {

        let wearables = Wearables.shared

        registrationTask?.cancel()
        registrationTask = Task {

            do {
                print("[Meta] Starting registration")
                try await Wearables.shared.startRegistration()
                print("[Meta] Registration completed successfully")
            } catch {
                print("[Meta] Registration failed:", error.localizedDescription)
            }

            // Surface the final registration state once registration settles.
            for await state in wearables.registrationStateStream() {
                if Task.isCancelled { return }
                logRegistrationState("registrationState", state)
            }
        }
    }

    // MARK: - Ray-Ban shared camera pipeline

    /// Starts a continuous Ray-Ban camera stream for use as a general camera
    /// source by the Tools pipeline. Each decoded frame is stored as the latest
    /// frame; JS polls `captureRayBanFrame` to feed it through the normal
    /// WebSocket frame pipeline. Resolves once the stream start is requested.
    @objc
    func startRayBanStream(
        _ resolve: @escaping RCTPromiseResolveBlock,
        rejecter reject: @escaping RCTPromiseRejectBlock
    ) {

        Task {

            do {

                // Don't create a new session until the previous one has fully
                // stopped, otherwise the SDK throws sessionAlreadyExists.
                if self.rayBanStopping {
                    reject(
                        "ray_ban_stopping",
                        "Ray-Ban session is still stopping, please try again",
                        nil
                    )
                    return
                }

                if self.rayBanSession != nil {
                    reject(
                        "ray_ban_already_active",
                        "Ray-Ban session is already active.",
                        nil
                    )
                    return
                }

                let wearables = Wearables.shared

                print("[Meta] Starting Ray-Ban camera stream")

                let permissionStatus = try await wearables.checkPermissionStatus(.camera)

                if String(describing: permissionStatus).lowercased() != "granted" {
                    let requestedStatus = try await wearables.requestPermission(.camera)

                    guard String(describing: requestedStatus).lowercased() == "granted" else {
                        reject(
                            "camera_permission_denied",
                            "Camera permission was not granted.",
                            nil
                        )
                        return
                    }
                }

                let deviceSelector = AutoDeviceSelector(wearables: wearables)
                try await self.waitForActiveDevice(deviceSelector)

                let session = try wearables.createSession(deviceSelector: deviceSelector)
                self.rayBanSession = session

                self.rayBanSessionStateToken = session.statePublisher.listen { [weak self] state in
                    print("[Meta] Ray-Ban session state:", String(describing: state))
                    self?.rayBanLastSessionState = String(describing: state)
                }

                self.rayBanSessionErrorToken = session.errorPublisher.listen { [weak self] error in
                    print("[Meta] Ray-Ban session error:", error.localizedDescription)
                    self?.rayBanLastSessionError = error.localizedDescription
                }

                try session.start()

                try await self.waitForSessionStarted(session)

                let config = StreamConfiguration(
                    videoCodec: .raw,
                    resolution: .high,
                    frameRate: 15
                )

                guard let stream = try session.addStream(config: config) else {
                    session.stop()
                    _ = await self.waitForRayBanSessionStopped(session)
                    self.releaseRayBanResources()
                    reject(
                        "stream_creation_failed",
                        "Unable to create a Meta DAT stream.",
                        nil
                    )
                    return
                }

                self.rayBanStream = stream

                self.rayBanStreamStateToken = stream.statePublisher.listen { [weak self] state in
                    print("[Meta] Ray-Ban stream state:", String(describing: state))
                    self?.rayBanLastStreamState = String(describing: state)
                }

                self.rayBanStreamErrorToken = stream.errorPublisher.listen { [weak self] error in
                    print("[Meta] Ray-Ban stream error:", String(describing: error), "-", error.localizedDescription)
                    self?.rayBanLastStreamError = "\(String(describing: error)) (\(error.localizedDescription))"
                }

                self.rayBanFrameToken = stream.videoFramePublisher.listen { [weak self] frame in
                    self?.handleRayBanFrame(frame)
                }

                await stream.start()
                print("[Meta] Ray-Ban stream started")

                resolve(true)

            } catch {
                print("[Meta] Failed to start Ray-Ban stream:", error.localizedDescription)

                // A partially created session/stream must be torn down here,
                // otherwise it lingers as rayBanSession/rayBanStream and every
                // retry is rejected with ray_ban_already_active.
                if let session = self.rayBanSession {
                    if let stream = self.rayBanStream {
                        await stream.stop()
                    }
                    session.stop()
                    _ = await self.waitForRayBanSessionStopped(session)
                }
                self.releaseRayBanResources()

                reject(
                    "start_ray_ban_stream_failed",
                    String(describing: error),
                    error
                )
            }
        }
    }

    /// Returns the most recent Ray-Ban frame as a JPEG data URI plus its
    /// dimensions, in the exact shape CameraView's phone path returns so tools
    /// cannot tell the two sources apart.
    @objc
    func captureRayBanFrame(
        _ resolve: @escaping RCTPromiseResolveBlock,
        rejecter reject: @escaping RCTPromiseRejectBlock
    ) {

        guard let image = self.latestRayBanImage else {
            reject(
                "no_ray_ban_frame",
                "No Ray-Ban frame is available yet. sessionState=\(self.rayBanLastSessionState) streamState=\(self.rayBanLastStreamState) framesReceived=\(self.rayBanFramesReceived) decodeFailed=\(self.rayBanFramesDecodeFailed) sessionError=\(self.rayBanLastSessionError ?? "none") streamError=\(self.rayBanLastStreamError ?? "none")",
                nil
            )
            return
        }

        guard let jpegData = image.jpegData(compressionQuality: 0.8) else {
            reject(
                "ray_ban_frame_encode_failed",
                "Failed to encode the latest Ray-Ban frame.",
                nil
            )
            return
        }

        let base64 = jpegData.base64EncodedString()

        resolve([
            "base64": "data:image/jpeg;base64,\(base64)",
            "width": Int(image.size.width),
            "height": Int(image.size.height),
        ])
    }

    /// Fully disconnects from the Ray-Ban camera: stops the stream, stops the
    /// parent DeviceSession, and — per Meta's lifecycle docs — only releases
    /// resources once SessionState reaches STOPPED. Resolves when the session
    /// has stopped (or after a safety timeout) so the app can switch back to
    /// the phone camera or start another tool without sessionAlreadyExists.
    @objc
    func stopRayBanStream(
        _ resolve: @escaping RCTPromiseResolveBlock,
        rejecter reject: @escaping RCTPromiseRejectBlock
    ) {

        Task {

            // Idempotent: a stop already in flight will release resources.
            if self.rayBanStopping {
                resolve(true)
                return
            }

            guard let session = self.rayBanSession else {
                // Nothing active — make sure state is clean and return.
                self.releaseRayBanResources()
                resolve(true)
                return
            }

            print("[Meta] Stopping Ray-Ban camera stream")
            self.rayBanStopping = true

            // 1. Stop the stream first (async), if one exists.
            if let stream = self.rayBanStream {
                await stream.stop()
            }

            // 2. Stop the parent DeviceSession.
            session.stop()

            // 3. Wait until the session actually reaches STOPPED before we
            //    release anything (resources must outlive the teardown).
            let reachedStopped = await self.waitForRayBanSessionStopped(session)

            if !reachedStopped {
                print("[Meta] Ray-Ban session did not reach STOPPED before timeout; releasing anyway")
            }

            // 4. Release Ray-Ban resources now that the session has stopped.
            self.releaseRayBanResources()
            self.rayBanStopping = false

            print("[Meta] Ray-Ban camera stream stopped")
            resolve(true)
        }
    }

    /// Polls the session state until it reports STOPPED. Returns false if
    /// STOPPED isn't reached within the safety timeout.
    private func waitForRayBanSessionStopped(_ session: DeviceSession) async -> Bool {

        for _ in 0..<50 {  // up to ~5 seconds at 100ms intervals

            if String(describing: session.state).lowercased().contains("stopped") {
                return true
            }

            try? await Task.sleep(nanoseconds: 100_000_000)
        }

        return false
    }

    /// Releases all Ray-Ban session/stream resources and listener tokens.
    /// Only call this after the session has reached STOPPED.
    private func releaseRayBanResources() {

        self.rayBanFrameToken = nil
        self.rayBanStreamStateToken = nil
        self.rayBanStreamErrorToken = nil
        self.rayBanSessionStateToken = nil
        self.rayBanSessionErrorToken = nil
        self.rayBanStream = nil
        self.rayBanSession = nil
        self.latestRayBanImage = nil
        self.rayBanFramesReceived = 0
        self.rayBanFramesDecodeFailed = 0
        self.rayBanLastStreamState = "none"
        self.rayBanLastSessionState = "none"
        self.rayBanLastSessionError = nil
        self.rayBanLastStreamError = nil
    }

    private func handleRayBanFrame(_ frame: VideoFrame) {

        self.rayBanFramesReceived += 1

        guard let image = frame.makeUIImage() else {
            self.rayBanFramesDecodeFailed += 1
            print("[Meta] Ray-Ban frame received but makeUIImage() returned nil")
            return
        }

        print("[Meta] Ray-Ban frame decoded:", image.size.width, "x", image.size.height)
        self.latestRayBanImage = image
    }

    // MARK: - Helpers

    /// Waits for `AutoDeviceSelector` to resolve a non-nil `activeDevice` before
    /// a session is created. AutoDeviceSelector resolves activeDevice
    /// asynchronously; calling createSession before it has settled throws
    /// noEligibleDevice even when a connected, compatible device exists. Gives
    /// up after a short timeout so it can never hang forever.
    private func waitForActiveDevice(_ selector: AutoDeviceSelector) async throws {

        if selector.activeDevice != nil {
            return
        }

        let streamTask = Task { () -> Bool in
            for await activeDevice in selector.activeDeviceStream() {
                if Task.isCancelled { return false }
                if activeDevice != nil { return true }
            }
            return false
        }

        let timeoutTask = Task {
            try? await Task.sleep(nanoseconds: 5_000_000_000)
            streamTask.cancel()
        }

        let resolved = await streamTask.value
        timeoutTask.cancel()

        guard resolved else {
            throw NSError(
                domain: "MetaWearablesModule",
                code: 1002,
                userInfo: [NSLocalizedDescriptionKey: "Timed out waiting for AutoDeviceSelector to resolve an active device."]
            )
        }
    }

    /// Polls the session state until it reports STARTED, throwing on timeout.
    private func waitForSessionStarted(_ session: DeviceSession) async throws {

        for _ in 0..<100 {  // up to ~20 seconds at 200ms intervals
            if String(describing: session.state).lowercased().contains("started") {
                return
            }
            try await Task.sleep(nanoseconds: 200_000_000)
        }

        throw NSError(
            domain: "MetaWearablesModule",
            code: 1001,
            userInfo: [NSLocalizedDescriptionKey: "Timed out waiting for the session to start."]
        )
    }

    /// Logs a registration state alongside its raw value so the numeric
    /// `registrationState` (e.g. 3 == registered) is always visible.
    private func logRegistrationState(_ label: String, _ state: Any) {

        if let rawRepresentable = state as? any RawRepresentable {
            print("[Meta] \(label): \(String(describing: state)) (raw \(String(describing: rawRepresentable.rawValue)))")
        } else {
            print("[Meta] \(label): \(String(describing: state))")
        }
    }
}
