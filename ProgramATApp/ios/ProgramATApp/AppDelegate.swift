import UIKit
import React
import React_RCTAppDelegate
import ReactAppDependencyProvider
import MWDATCore

@main
class AppDelegate: UIResponder, UIApplicationDelegate {
  var window: UIWindow?

  var reactNativeDelegate: ReactNativeDelegate?
  var reactNativeFactory: RCTReactNativeFactory?

  func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
  ) -> Bool {
    do {
      try Wearables.configure()
      print("Wearables configured")
    } catch {
      print("Wearables configure failed: \(error)")
    }

    let delegate = ReactNativeDelegate()
    let factory = RCTReactNativeFactory(delegate: delegate)
    delegate.dependencyProvider = RCTAppDependencyProvider()

    reactNativeDelegate = delegate
    reactNativeFactory = factory

    window = UIWindow(frame: UIScreen.main.bounds)

    factory.startReactNative(
      withModuleName: "ProgramATApp",
      in: window,
      launchOptions: launchOptions
    )

    return true
  }

  func application(
    _ app: UIApplication,
    open url: URL,
    options: [UIApplication.OpenURLOptionsKey : Any] = [:]
  ) -> Bool {
    print("URL CALLBACK RECEIVED:", url.absoluteString)

    Task {
      do {
        print("CALLING HANDLE URL")
        _ = try await Wearables.shared.handleUrl(url)
        print("HANDLE URL SUCCESS from open url")
      } catch {
        print("HANDLE URL FAILED from open url:", error)
        print("LOCALIZED:", error.localizedDescription)
        print("MIRROR:", Mirror(reflecting: error))
      }
    }

    return true
  }

  func application(
    _ application: UIApplication,
    continue userActivity: NSUserActivity,
    restorationHandler: @escaping ([UIUserActivityRestoring]?) -> Void
  ) -> Bool {
    print("USER ACTIVITY RECEIVED:", userActivity.activityType)

    guard userActivity.activityType == NSUserActivityTypeBrowsingWeb,
          let url = userActivity.webpageURL else {
      print("USER ACTIVITY did not contain webpageURL")
      return false
    }

    print("UNIVERSAL LINK CALLBACK RECEIVED:", url.absoluteString)

    Task {
      do {
        _ = try await Wearables.shared.handleUrl(url)
        print("HANDLE URL SUCCESS from universal link")
      } catch {
        print("HANDLE URL FAILED from universal link:", error)
        print("LOCALIZED:", error.localizedDescription)
        print("MIRROR:", Mirror(reflecting: error))
      }
    }

    return true
  }
}

class ReactNativeDelegate: RCTDefaultReactNativeFactoryDelegate {
  override func sourceURL(for bridge: RCTBridge) -> URL? {
    self.bundleURL()
  }

  override func bundleURL() -> URL? {
#if DEBUG
    RCTBundleURLProvider.sharedSettings().jsBundleURL(forBundleRoot: "index")
#else
    Bundle.main.url(forResource: "main", withExtension: "jsbundle")
#endif
  }
}
