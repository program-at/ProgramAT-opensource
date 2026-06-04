//
//  MetaWearablesModule.m
//  ProgramATApp
//
//  Created by jxr on 5/28/26.
//

#import <Foundation/Foundation.h>
#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(MetaWearablesModule, NSObject)

RCT_EXTERN_METHOD(registerDevice)
RCT_EXTERN_METHOD(startRayBanStream:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(captureRayBanFrame:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(stopRayBanStream:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)

@end
