//
//  MetaWearablesModule.m
//  ProgramATApp
//
//  Created by jxr on 5/28/26.
//

#import <Foundation/Foundation.h>
#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(MetaWearablesModule, NSObject)

RCT_EXTERN_METHOD(hello)
RCT_EXTERN_METHOD(createMockDevice)
RCT_EXTERN_METHOD(useBackCameraFeed)
RCT_EXTERN_METHOD(requestDoorRecognitionTest:(NSString *)backendURLString)
RCT_EXTERN_METHOD(startMockCameraStream)
RCT_EXTERN_METHOD(startFirstFrameCapture:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(listDevices)
RCT_EXTERN_METHOD(listDevicesNow)
RCT_EXTERN_METHOD(debugWearablesState)

@end
