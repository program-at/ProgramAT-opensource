//
//  MetaWearableModule.swift
//  ProgramATApp
//
//  Created by jxr on 5/28/26.
//

import Foundation
import React
import MWDATCore
import MWDATMockDevice

@objc(MetaWearablesModule)
class MetaWearablesModule: NSObject {

    private var mockDevice: (any MockRaybanMeta)?

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
    static func requiresMainQueueSetup() -> Bool {
        return false
    }
}
