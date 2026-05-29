//
//  MetaWearableModule.swift
//  ProgramATApp
//
//  Created by jxr on 5/28/26.
//

import Foundation
import React
import MWDATCore
import MWDATCamera
import MWDATMockDevice

@objc(MetaWearablesModule)
class MetaWearablesModule: NSObject {

    private var mockDevice: (any MockRaybanMeta)?
    private var session: DeviceSession?
    private var stream: MWDATCamera.Stream?
    private var sessionStateToken: Any?
    private var sessionErrorToken: Any?
    private var streamStateToken: Any?
    private var streamErrorToken: Any?
    private var frameToken: Any?

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

        print("Mock device ready")
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
    func startMockCameraStream() {

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

                        self.logFrame(frame)

                    }

                await stream.start()

                print("Stream current state after start():", stream.state)

                print("Stream start requested successfully")

                Task {
                    try? await Task.sleep(for: .seconds(2))

                    if let session = self.session {
                        print("SESSION STATE AFTER 2s:", session.state)
                    }

                    if let stream = self.stream {
                        print("STREAM STATE AFTER 2s:", stream.state)
                    }
                }

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

}
