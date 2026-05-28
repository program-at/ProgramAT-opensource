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

                let selector = AutoDeviceSelector(
                    wearables: wearables
                )

                let session = try wearables.createSession(
                    deviceSelector: selector
                )

                self.session = session

                print("Session created")

                let _ = session.statePublisher.listen { state in
                    print("SESSION STATE:", state)
                }

                let _ = session.errorPublisher.listen { error in
                    print("SESSION ERROR:", error)
                }

                try session.start()

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

                print("Stream created")

                let _ = stream.statePublisher.listen { state in
                    print("STREAM STATE:", state)
                }

                let _ = stream.errorPublisher.listen { error in
                    print("STREAM ERROR:", error)
                }

                let _ = stream.videoFramePublisher.listen { frame in

                    print("FRAME RECEIVED")

                    self.logFrame(frame)

                }

                await stream.start()

                print("Stream start requested")

                print("Calling listDevices after stream start")
                self.listDevices()
                print("listDevices call finished")

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

        let wearables = Wearables.shared

        let devices = wearables.devices

        NSLog("DEVICES COUNT: %d", devices.count)

        for device in devices {
            NSLog("%@", String(describing: device))
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
