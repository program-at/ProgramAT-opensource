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
    private var uploadedTestFrame = false
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
                        print("SESSION STATE:", state)
                    }

                sessionErrorToken =
                    session.errorPublisher.listen { error in
                        print("SESSION ERROR:", error)
                    }

                try session.start()

                print("Session current state after start():", session.state)

                print("Session start requested")

                let config = StreamConfiguration(
                    videoCodec: .raw,
                    resolution: .low,
                    frameRate: 24
                )

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
                        print("STREAM STATE:", state)
                    }

                streamErrorToken =
                    stream.errorPublisher.listen { error in
                        print("STREAM ERROR:", error)
                    }

                frameToken =
                    stream.videoFramePublisher.listen { frame in

                        print("FRAME RECEIVED")

                        guard self.uploadedTestFrame == false else {
                            return
                        }

                        self.uploadedTestFrame = true
                        self.processTestFrame(frame)

                    }

                await stream.start()

                print("Stream current state after start():", stream.state)

                print("Stream start requested successfully")

                print("Calling listDevicesNow after stream start")
                self.listDevicesNow()
                print("listDevicesNow call finished")

            } catch {

                print(
                    "startMockCameraStream failed:",
                    error
                )

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

                    guard let device =
                        wearables.deviceForIdentifier(deviceIdentifier)
                    else {
                        continue
                    }

                    print(
                        "Device compatibility:",
                        device.compatibility()
                    )

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

    @objc
    static func requiresMainQueueSetup() -> Bool {
        return false
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

    private func processTestFrame(_ frame: Any) {

        guard let videoFrame = frame as? VideoFrame else {
            print("FRAME TYPE NOT SUPPORTED:", type(of: frame))
            return
        }

        guard let image = videoFrame.makeUIImage() else {
            print("Failed to create UIImage from VideoFrame")
            return
        }

        guard let jpegData = image.jpegData(compressionQuality: 0.9) else {
            print("Failed to create JPEG data")
            return
        }

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

                print("JPEG saved successfully")
                print("image width:", image.size.width)
                print("image height:", image.size.height)
                print("jpeg size:", jpegData.count)
                print("saved path:", placeholderIdentifier)

                if self.pendingDoorRecognitionTest {
                    self.pendingDoorRecognitionTest = false
                    self.uploadTestFrameToBackend(jpegData)
                }
            })
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
        }.resume()
    }

}
