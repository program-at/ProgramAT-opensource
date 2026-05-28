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

    @objc
    func hello() {
        print("Hello bridge works")
    }

    @objc
    func createMockDevice() {
        let device = MockDeviceKit.shared.pairRaybanMeta()

        if device != nil {
            print("Mock device created")
        } else {
            print("Failed to create mock device")
        }
    }

    @objc
    static func requiresMainQueueSetup() -> Bool {
        return false
    }
}
