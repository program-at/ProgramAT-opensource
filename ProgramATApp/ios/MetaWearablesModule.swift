//
//  MetaWearableModule.swift
//  ProgramATApp
//
//  Created by jxr on 5/28/26.
//

import Foundation
import React
import MWDATCore
@objc(MetaWearablesModule)
class MetaWearablesModule: NSObject {
  
  @objc
  func hello() {
    print("Meta SDK connected")
  }
  
  @objc
  static func requiresMainQueueSetup() -> Bool {
    return false
  }
}

