//
//  MetaWearableModule.swift
//  ProgramATApp
//
//  Created by jxr on 5/28/26.
//

import Foundation
import React
import UIKit
import Photos
import MWDATCore
import MWDATCamera
import MWDATMockDevice

@objc(MetaWearablesModule)
class MetaWearablesModule: NSObject {

    private var saveFramesToPhotos = false
    private var saveFramesToDocuments = false
    private var uploadFramesToBackend = true

    private var frameCount = 0
    private var didLogFirstFrame = false

    private var mockDevice: (any MockRaybanMeta)?
    private var mockDeviceReady = false
    private var pendingMockStreamStart = false
    private var mockDeviceRegistrationTask: Task<Void, Never>?
    private var session: DeviceSession?
    private var stream: MWDATCamera.Stream?
    private var sessionStateToken: Any?
    private var sessionErrorToken: Any?
    private var streamStateToken: Any?
    private var streamErrorToken: Any?
    private var frameToken: Any?
    private var physicalRegistrationTask: Task<Void, Never>?
    private var physicalDevicesTask: Task<Void, Never>?
    private var physicalSession: DeviceSession?
    private var physicalStream: MWDATCamera.Stream?
    private var physicalSessionStateToken: Any?
    private var physicalSessionErrorToken: Any?
    private var physicalStreamStateToken: Any?
    private var physicalStreamErrorToken: Any?
    private var physicalFrameToken: Any?
    private var physicalCaptureResolved = false
    private var registrationDebugTask: Task<Void, Never>?
    private var pendingDoorRecognitionTest = false
    private var pendingDoorRecognitionBackendURL: URL?

    @objc
    func hello() {
        print("Hello bridge works")
    }

    @objc
    func createMockDevice() {

        MockDeviceKit.shared.enable()

        let device = MockDeviceKit.shared.pairRaybanMeta()

        device.powerOn()
        device.don()
        device.unfold()

        mockDevice = device

        mockDeviceReady = false

        observeMockDeviceRegistration()

        print("Mock device created; waiting for registration...")
    }

    @objc
    func useBackCameraFeed() {

        guard let device = mockDevice else {
            print("No mock device")
            return
        }

        Task {
            await device.services.camera.setCameraFeed(
                cameraFacing: .back
            )

            print("Back camera feed configured")
        }
    }

    @objc
    func requestDoorRecognitionTest(_ backendURLString: String) {

        let trimmed = backendURLString.trimmingCharacters(in: .whitespacesAndNewlines)

        guard let backendURL = URL(string: trimmed) else {
            print("Invalid backend URL for door recognition test:", backendURLString)
            return
        }

        pendingDoorRecognitionBackendURL = backendURL
        pendingDoorRecognitionTest = true

        print("Door recognition backend test armed:", backendURL.absoluteString)
    }

    @objc
    func startMockCameraStream() {

        guard mockDeviceReady else {
            pendingMockStreamStart = true
            print("Mock device not ready yet")
            return
        }

        startMockCameraStreamInternal()
    }

    private func startMockCameraStreamInternal() {

        Task {

            do {

                let wearables = Wearables.shared

                print("Preparing to create stream session")

                guard let selectedDeviceIdentifier = wearables.devices.first else {
                    print("No available device identifier found")
                    return
                }

                print("selectedDeviceIdentifier:", selectedDeviceIdentifier)

                let selector = SpecificDeviceSelector(
                    device: selectedDeviceIdentifier
                )

                print("Using SpecificDeviceSelector for selected device")

                let session = try wearables.createSession(
                    deviceSelector: selector
                )

                self.session = session

                print("Session current state immediately after creation:", session.state)

                print("Session created successfully")

                sessionStateToken =
                    session.statePublisher.listen { state in
                        self.logWearablesState("SESSION STATE", state)
                    }

                sessionErrorToken =
                    session.errorPublisher.listen { error in
                        self.logWearablesError("SESSION ERROR", error)
                    }

                try session.start()

                print("Session current state after start():", session.state)

                print("Session start requested")

                self.logConnectedDeviceInfo(
                    deviceIdentifier: selectedDeviceIdentifier
                )

                let config = StreamConfiguration(
                    videoCodec: .raw,
                    resolution: .high,
                    frameRate: 24
                )

                print("Stream resolution:", String(describing: config.resolution))
                print("Stream frame rate:", config.frameRate)
                print("Stream codec:", String(describing: config.videoCodec))

                guard let stream = try session.addStream(
                    config: config
                ) else {

                    print("Failed to create stream")
                    return
                }

                self.stream = stream

                print("Stream current state immediately after creation:", stream.state)

                print("Stream created successfully")

                streamStateToken =
                    stream.statePublisher.listen { state in
                        self.logWearablesState("STREAM STATE", state)
                    }

                streamErrorToken =
                    stream.errorPublisher.listen { error in
                        self.logWearablesError("STREAM ERROR", error)
                    }

                frameToken =
                    stream.videoFramePublisher.listen { frame in
                        self.handleIncomingFrame(frame)

                    }

                await stream.start()

                print("Stream current state after start():", stream.state)

                print("Stream start requested successfully")

                print("Calling listDevicesNow after stream start")
                self.listDevicesNow()
                print("listDevicesNow call finished")

            } catch {

                self.logWearablesError("startMockCameraStream failed", error)

            }
        }
    }

    @objc
    func startFirstFrameCapture(
        _ resolve: @escaping RCTPromiseResolveBlock,
        rejecter reject: @escaping RCTPromiseRejectBlock
    ) {

        Task {

            do {

                let wearables = Wearables.shared

                physicalCaptureResolved = false

                print("Starting physical device smoke test")
                self.logRegistrationState("registrationState", wearables.registrationState)

                physicalRegistrationTask?.cancel()
                physicalRegistrationTask = Task { [weak self] in

                    guard let self else { return }

                    var lastRawValue: String?

                    for await state in wearables.registrationStateStream() {

                        if Task.isCancelled {
                            return
                        }

                        let rawValue = (state as? any RawRepresentable)
                            .map { String(describing: $0.rawValue) }

                        // Only log when the state actually changes.
                        if rawValue != lastRawValue {
                            lastRawValue = rawValue
                            self.logRegistrationState("registrationState changed", state)
                        }
                    }
                }

                physicalDevicesTask?.cancel()
                physicalDevicesTask = Task { [weak self] in

                    guard let self else { return }

                    for await devices in wearables.devicesStream() {

                        if Task.isCancelled {
                            return
                        }

                        print("DEVICES STREAM count:", devices.count)

                        for deviceIdentifier in devices {
                            self.logDeviceDetails("devicesStream", deviceIdentifier)
                        }
                    }
                }

                let permissionStatus = try await wearables.checkPermissionStatus(.camera)
                print("Camera permission status:", permissionStatus)

                if String(describing: permissionStatus).lowercased() != "granted" {
                    let requestedStatus = try await wearables.requestPermission(.camera)
                    print("Camera permission requested:", requestedStatus)

                    guard String(describing: requestedStatus).lowercased() == "granted" else {
                        reject(
                            "camera_permission_denied",
                            "Camera permission was not granted.",
                            nil
                        )
                        return
                    }
                }

                // Diagnostics: dump every device DAT knows about and explain
                // which one AutoDeviceSelector is likely to pick (or why none
                // qualify) right before we attempt to create the session.
                self.logPreSessionDiagnostics(wearables)

                let deviceSelector = AutoDeviceSelector(wearables: wearables)
                print("AutoDeviceSelector.activeDevice (immediately after init):",
                      String(describing: deviceSelector.activeDevice))

                // AutoDeviceSelector resolves activeDevice asynchronously. If we
                // call createSession before it has settled it throws
                // noEligibleDevice even though wearables.devices already holds a
                // connected, compatible device. Wait for activeDevice to become
                // non-nil (bounded by a timeout) before creating the session.
                try await self.waitForActiveDevice(deviceSelector)

                print("Creating session with selector: AutoDeviceSelector(wearables:), activeDevice:",
                      String(describing: deviceSelector.activeDevice))
                let session = try wearables.createSession(
                    deviceSelector: deviceSelector
                )

                physicalSession = session

                physicalSessionStateToken = session.statePublisher.listen { state in
                    self.logWearablesState("PHYSICAL SESSION STATE", state)
                }

                physicalSessionErrorToken = session.errorPublisher.listen { error in
                    self.logWearablesError("PHYSICAL SESSION ERROR", error)
                }

                try session.start()
                print("Physical session start requested")

                try await waitForPhysicalSessionStart(session)

                let config = StreamConfiguration(
                    videoCodec: .raw,
                    resolution: .low,
                    frameRate: 15
                )

                print("Physical stream resolution:", String(describing: config.resolution))
                print("Physical stream frame rate:", config.frameRate)
                print("Physical stream codec:", String(describing: config.videoCodec))

                guard let stream = try session.addStream(config: config) else {
                    reject(
                        "stream_creation_failed",
                        "Unable to create a Meta DAT stream.",
                        nil
                    )
                    return
                }

                physicalStream = stream

                physicalStreamStateToken = stream.statePublisher.listen { state in
                    self.logWearablesState("PHYSICAL STREAM STATE", state)
                }

                physicalStreamErrorToken = stream.errorPublisher.listen { error in
                    self.logWearablesError("PHYSICAL STREAM ERROR", error)
                }

                physicalFrameToken = stream.videoFramePublisher.listen { [weak self] frame in
                    self?.handleFirstPhysicalFrame(
                        frame,
                        resolve: resolve,
                        reject: reject
                    )
                }

                await stream.start()
                print("Physical stream start requested")

            } catch {
                self.logWearablesError("startFirstFrameCapture failed", error)
                reject(
                    "start_first_frame_capture_failed",
                    String(describing: error),
                    error
                )
            }
        }
    }

    @objc
    func debugRegistration() {

        let wearables = Wearables.shared

        logDatRuntimeConfig()

        registrationDebugTask?.cancel()

        registrationDebugTask = Task {

            do {
                print("STARTING REGISTRATION")
                try await Wearables.shared.startRegistration()
                print("startRegistration returned successfully")
            } catch {
                logWearablesError("startRegistration failed", error)
            }

            for await state in wearables.registrationStateStream() {
                if Task.isCancelled {
                    return
                }

                self.logRegistrationState("REGISTRATION STATE STREAM", state)
            }
        }
    }

    private func observeMockDeviceRegistration() {

        mockDeviceRegistrationTask?.cancel()

        mockDeviceRegistrationTask = Task { [weak self] in

            guard let self else { return }

            let wearables = Wearables.shared

            print("Waiting for mock device registration...")

            for await deviceIdentifiers in wearables.devicesStream() {

                if Task.isCancelled {
                    return
                }

                print("Device count:", deviceIdentifiers.count)

                for deviceIdentifier in deviceIdentifiers {

                    self.logDeviceDetails("mock devicesStream", deviceIdentifier)

                    guard let device =
                        wearables.deviceForIdentifier(deviceIdentifier)
                    else {
                        continue
                    }

                    guard String(describing: device.compatibility()) == "compatible" else {
                        continue
                    }

                    print("Mock device discovered")
                    print("Device identifier:", deviceIdentifier)

                    self.mockDeviceReady = true

                    self.mockDeviceRegistrationTask?.cancel()

                    if self.pendingMockStreamStart {
                        self.pendingMockStreamStart = false
                        self.startMockCameraStream()
                    }

                    return
                }
            }
        }
    }

    @objc
    func listDevices() {

        listDevicesNow()
    }

    @objc
    static func requiresMainQueueSetup() -> Bool {
        return false
    }

    private func handleIncomingFrame(_ frame: VideoFrame) {

        frameCount += 1

        if didLogFirstFrame == false {
            didLogFirstFrame = true
            print("First frame received")
        } else if frameCount % 30 == 0 {
            print("Frame count:", frameCount)
        }

        guard saveFramesToPhotos || saveFramesToDocuments || uploadFramesToBackend else {
            return
        }

        guard let jpegData = makeJPEGData(from: frame) else {
            print("Failed to create JPEG data")
            return
        }

        if saveFramesToPhotos || saveFramesToDocuments {
            saveTestImage(jpegData)
        }

        if uploadFramesToBackend {
            uploadTestFrameToBackend(jpegData)
        }
    }

    private func makeJPEGData(from videoFrame: VideoFrame) -> Data? {

        guard let image = videoFrame.makeUIImage() else {
            print("Failed to create UIImage from VideoFrame")
            return nil
        }

        return image.jpegData(compressionQuality: 0.9)
    }

    private func logConnectedDeviceInfo(deviceIdentifier: String) {

        let wearables = Wearables.shared

        guard let device = wearables.deviceForIdentifier(deviceIdentifier) else {
            print("[Meta] Connected device:")
            print("id=\(deviceIdentifier)")
            print("name=<unknown>")
            print("type=<unknown>")
            print("linkState=<unknown>")
            return
        }

        print("[Meta] Connected device:")
        print("id=\(deviceIdentifier)")
        print("name=\(device.nameOrId())")
        print("type=\(device.deviceType())")
        print("linkState=\(device.linkState)")
    }

    /// Waits for `AutoDeviceSelector` to resolve a non-nil `activeDevice`
    /// before a session is created. Prints every `activeDeviceStream()` update
    /// and gives up after a short timeout so it can never hang forever.
    private func waitForActiveDevice(_ selector: AutoDeviceSelector) async throws {

        if let activeDevice = selector.activeDevice {
            print("AutoDeviceSelector.activeDevice already resolved:", String(describing: activeDevice))
            return
        }

        print("AutoDeviceSelector.activeDevice is nil; waiting up to 5s for activeDeviceStream()...")

        let streamTask = Task { () -> Bool in

            for await activeDevice in selector.activeDeviceStream() {

                if Task.isCancelled {
                    return false
                }

                print("activeDeviceStream update:", String(describing: activeDevice))

                if activeDevice != nil {
                    return true
                }
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

        print("AutoDeviceSelector.activeDevice resolved:", String(describing: selector.activeDevice))
    }

    private func waitForPhysicalSessionStart(_ session: DeviceSession) async throws {

        for _ in 0..<100 {
            let stateDescription = String(describing: session.state)
            print("PHYSICAL SESSION POLL STATE:", stateDescription)

            if stateDescription.lowercased().contains("started") {
                return
            }

            try await Task.sleep(nanoseconds: 200_000_000)
        }

        throw NSError(
            domain: "MetaWearablesModule",
            code: 1001,
            userInfo: [NSLocalizedDescriptionKey: "Timed out waiting for the physical session to start."]
        )
    }

    private func logDatRuntimeConfig() {

        let infoDictionary = Bundle.main.infoDictionary ?? [:]

        print("========== DAT CONFIG ==========")
        print("CFBundleURLTypes:", infoDictionary["CFBundleURLTypes"] ?? "<missing>")

        if let mwdat = infoDictionary["MWDAT"] as? [String: Any] {
            print("MWDAT.AppLinkURLScheme:", mwdat["AppLinkURLScheme"] ?? "<missing>")
            print("MWDAT.MetaAppID:", mwdat["MetaAppID"] ?? "<missing>")
            print("MWDAT.ClientToken:", mwdat["ClientToken"] ?? "<missing>")
            print("MWDAT.TeamID:", mwdat["TeamID"] ?? "<missing>")
            print("MWDAT.DAMEnabled:", mwdat["DAMEnabled"] ?? "<missing>")
        } else {
            print("MWDAT:", "<missing>")
        }
    }

    private func logWearablesState(_ label: String, _ state: Any) {

        print("\(label): STATE = \(String(describing: state))")
        print("\(label): MIRROR = \(Mirror(reflecting: state))")

        if let rawRepresentable = state as? any RawRepresentable {
            print("\(label): RAW VALUE = \(String(describing: rawRepresentable.rawValue))")
        }

        let mirror = Mirror(reflecting: state)
        if mirror.children.isEmpty == false {
            for (index, child) in mirror.children.enumerated() {
                let childLabel = child.label ?? "<unlabeled>"
                print("\(label): CHILD[\(index)] \(childLabel) = \(String(describing: child.value))")
            }
        }
    }

    private func logWearablesError(_ label: String, _ error: Error) {

        print("\(label): error = \(error)")
        print("\(label): String(describing:) = \(String(describing: error))")
        print("\(label): error.localizedDescription = \(error.localizedDescription)")
        print("\(label): error.mirror = \(Mirror(reflecting: error))")

        let mirror = Mirror(reflecting: error)
        if mirror.children.isEmpty == false {
            for (index, child) in mirror.children.enumerated() {
                let childLabel = child.label ?? "<unlabeled>"
                print("\(label): error.child[\(index)] \(childLabel) = \(String(describing: child.value))")
            }
        }
    }

    /// Standardized, detailed dump of a single DAT device. Used everywhere
    /// `devicesStream()` (or `wearables.devices`) is observed so we can see
    /// exactly what DAT considers eligible for a session.
    private func logDeviceDetails(_ context: String, _ deviceIdentifier: String) {

        let wearables = Wearables.shared

        print("========== DEVICE ==========")
        print("context: \(context)")
        print("id: \(deviceIdentifier)")

        guard let device = wearables.deviceForIdentifier(deviceIdentifier) else {
            print("type: <device lookup failed>")
            print("linkState: <unknown>")
            print("compatibility: <unknown>")
            print("displayName: <unknown>")
            print("============================")
            return
        }

        print("type: \(String(describing: device.deviceType()))")
        print("linkState: \(String(describing: device.linkState))")
        print("compatibility: \(String(describing: device.compatibility()))")
        print("displayName: \(device.nameOrId())")

        let mirror = Mirror(reflecting: device)
        for child in mirror.children {
            if let childLabel = child.label {
                print("prop \(childLabel): \(String(describing: child.value))")
            }
        }

        print("============================")
    }

    /// Logs a registration state alongside its raw value so the numeric
    /// `registrationState` (e.g. 3 == registered) is always visible.
    private func logRegistrationState(_ label: String, _ state: Any) {

        print("\(label): state = \(String(describing: state))")

        if let rawRepresentable = state as? any RawRepresentable {
            print("\(label): rawValue = \(String(describing: rawRepresentable.rawValue))")
        } else {
            print("\(label): rawValue = <not RawRepresentable>")
        }
    }

    /// Logs everything DAT knows right before we ask `AutoDeviceSelector` to
    /// pick a device, including which device the selector is likely to choose
    /// and why each candidate does/doesn't qualify.
    private func logPreSessionDiagnostics(_ wearables: any WearablesInterface) {

        print("========== PRE-SESSION DIAGNOSTICS ==========")
        print("selector: AutoDeviceSelector(wearables:)")

        logRegistrationState("registrationState", wearables.registrationState)

        let deviceIds = wearables.devices
        print("available device count: \(deviceIds.count)")

        var likelySelectedId: String?

        for deviceId in deviceIds {

            logDeviceDetails("pre-session", deviceId)

            guard let device = wearables.deviceForIdentifier(deviceId) else {
                print("MATCH \(deviceId): device lookup failed -> NOT eligible")
                continue
            }

            let compatibility = String(describing: device.compatibility())
            let linkState = String(describing: device.linkState)
            let isCompatible = compatibility.lowercased() == "compatible"

            print("MATCH \(deviceId): compatibility=\(compatibility) linkState=\(linkState) compatible=\(isCompatible)")

            if isCompatible && likelySelectedId == nil {
                likelySelectedId = deviceId
            }
        }

        if let likelySelectedId {
            print("AutoDeviceSelector would likely select: \(likelySelectedId)")
        } else {
            print("AutoDeviceSelector would find NO eligible device (none compatible)")
        }

        print("=============================================")
    }

    private func handleFirstPhysicalFrame(
        _ frame: VideoFrame,
        resolve: @escaping RCTPromiseResolveBlock,
        reject: @escaping RCTPromiseRejectBlock
    ) {

        guard physicalCaptureResolved == false else {
            return
        }

        guard let image = frame.makeUIImage() else {
            print("Failed to create UIImage from physical frame")
            return
        }

        guard let jpegData = image.jpegData(compressionQuality: 0.9) else {
            print("Failed to convert physical frame to JPEG data")
            return
        }

        let documentsDirectory = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first

        guard let documentsDirectory else {
            reject(
                "documents_directory_missing",
                "Failed to locate the Documents directory.",
                nil
            )
            return
        }

        let fileURL = documentsDirectory.appendingPathComponent("frame.jpg")

        do {
            try jpegData.write(to: fileURL, options: .atomic)
            physicalCaptureResolved = true
            print("FRAME RECEIVED")
            print("Saved frame path:", fileURL.path)
            resolve(fileURL.path)
        } catch {
            print("Failed to save physical frame:", error)
            reject(
                "frame_save_failed",
                String(describing: error),
                error
            )
        }
    }

    // MARK: - Debug / Testing

    @objc
    func listDevicesNow() {

        let wearables = Wearables.shared

        let devices = wearables.devices

        NSLog("DEVICES COUNT: %d", devices.count)

        for device in devices {
            NSLog("%@", String(describing: device))
        }
    }

    @objc
    func debugWearablesState() {

        let wearables = Wearables.shared

        print("========== DEBUG ==========")

        print("registrationState:", wearables.registrationState)

        print("deviceCount:", wearables.devices.count)

        for id in wearables.devices {

            print("device id:", id)

            guard let device =
                wearables.deviceForIdentifier(id)
            else {
                print("device nil")
                continue
            }

            print("name:", device.nameOrId())
            print("linkState:", device.linkState)
            print("compatibility:", device.compatibility())
            print("deviceType:", device.deviceType())
        }
    }

    private func logFrame(_ frame: Any) {

        print("FRAME TYPE:", type(of: frame))

        let mirror = Mirror(reflecting: frame)

        print(
            "FRAME PROPERTIES:",
            mirror.children.count
        )

        for child in mirror.children {

            if let label = child.label {

                print(
                    "  \(label):",
                    type(of: child.value)
                )

            }
        }
    }


    private func saveTestImage(_ jpegData: Data) {

        if saveFramesToPhotos {
            saveTestImageToPhotos(jpegData)
        }

        if saveFramesToDocuments {
            saveTestImageToDocuments(jpegData)
        }
    }

    private func saveTestImageToPhotos(_ jpegData: Data) {

        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in

            guard status == .authorized || status == .limited else {
                print("Failed to save to photo library: authorization status =", status.rawValue)
                return
            }

            var placeholderIdentifier = ""

            PHPhotoLibrary.shared().performChanges({

                let request = PHAssetCreationRequest.forAsset()
                let options = PHAssetResourceCreationOptions()
                options.uniformTypeIdentifier = "public.jpeg"
                request.addResource(with: .photo, data: jpegData, options: options)
                placeholderIdentifier = request.placeholderForCreatedAsset?.localIdentifier ?? ""

            }, completionHandler: { success, error in

                if let error {
                    print("Failed to save to photo library:", error)
                    return
                }

                guard success else {
                    print("Failed to save to photo library")
                    return
                }

                print("JPEG saved successfully to photo library")
                print("saved path:", placeholderIdentifier)
            })
        }
    }

    private func saveTestImageToDocuments(_ jpegData: Data) {

        let documentsDirectory = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first

        guard let documentsDirectory else {
            print("Failed to locate Documents directory")
            return
        }

        let formatter = ISO8601DateFormatter()
        let filename = "meta-frame-\(formatter.string(from: Date())).jpg"
        let fileURL = documentsDirectory.appendingPathComponent(filename)

        do {
            try jpegData.write(to: fileURL, options: .atomic)
            print("JPEG saved successfully to Documents:", fileURL.lastPathComponent)
        } catch {
            print("Failed to save to Documents:", error)
        }
    }

    private func uploadTestFrameToBackend(_ jpegData: Data) {

        guard let backendURL = pendingDoorRecognitionBackendURL else {
            print("No backend URL set for door recognition test")
            return
        }

        let testURL = backendURL.appendingPathComponent("test-door-recognition")

        print("Uploading Meta frame to backend:", testURL.absoluteString)

        var request = URLRequest(url: testURL)
        request.httpMethod = "POST"
        request.setValue("image/jpeg", forHTTPHeaderField: "Content-Type")

        URLSession.shared.uploadTask(with: request, from: jpegData) { data, response, error in

            if let error {
                print("Failed to upload Meta frame to backend:", error)
                return
            }

            if let httpResponse = response as? HTTPURLResponse {
                print("Door recognition backend status:", httpResponse.statusCode)
            }

            if let data,
               let responseText = String(data: data, encoding: .utf8) {
                print("Door recognition backend response:", responseText)
            }

            if self.pendingDoorRecognitionTest {
                self.pendingDoorRecognitionTest = false
            }
        }.resume()
    }

}
