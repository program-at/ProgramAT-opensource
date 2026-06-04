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
    print("🚨🚨🚨 DID FINISH LAUNCHING 🚨🚨🚨")
    print("launchOptions =", String(describing: launchOptions))

    if let url = launchOptions?[.url] as? URL {
      print("launchOptions.url =", url.absoluteString)
    }

    if let userActivityDictionary = launchOptions?[.userActivityDictionary] as? [AnyHashable: Any] {
      print("launchOptions.userActivityDictionary =", userActivityDictionary)
      for (key, value) in userActivityDictionary {
        print("launchOptions.userActivityDictionary[\(key)] =", value)
        if let userActivity = value as? NSUserActivity {
          print("launchOptions.userActivity.activityType =", userActivity.activityType)
          print("launchOptions.userActivity.webpageURL =", String(describing: userActivity.webpageURL?.absoluteString))
        }
      }
    }

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

  func applicationDidBecomeActive(_ application: UIApplication) {
    print("🚨🚨🚨 APP DID BECOME ACTIVE 🚨🚨🚨")
  }

  func applicationWillEnterForeground(_ application: UIApplication) {
    print("🚨🚨🚨 APP WILL ENTER FOREGROUND 🚨🚨🚨")
  }

  func application(
    _ app: UIApplication,
    open url: URL,
    options: [UIApplication.OpenURLOptionsKey : Any] = [:]
  ) -> Bool {
    print("🚨🚨🚨 OPEN URL CALLBACK FIRED 🚨🚨🚨")
    print("URL =", url.absoluteString)

    Task {
      do {
        print("CALLING Wearables.shared.handleUrl()")
        _ = try await Wearables.shared.handleUrl(url)
        print("HANDLE URL SUCCESS")
      } catch {
        print("HANDLE URL FAILURE")
        print(error)
        print(error.localizedDescription)
        print(Mirror(reflecting: error))
      }
    }

    return true
  }

  func application(
    _ application: UIApplication,
    continue userActivity: NSUserActivity,
    restorationHandler: @escaping ([UIUserActivityRestoring]?) -> Void
  ) -> Bool {
    print("🚨🚨🚨 UNIVERSAL LINK CALLBACK FIRED 🚨🚨🚨")
    print("activityType =", userActivity.activityType)

    guard userActivity.activityType == NSUserActivityTypeBrowsingWeb,
          let url = userActivity.webpageURL else {
      print("USER ACTIVITY did not contain webpageURL")
      return false
    }

    print("WEBPAGE URL =", url.absoluteString)

    Task {
      do {
        print("CALLING handleUrl FROM UNIVERSAL LINK")
        _ = try await Wearables.shared.handleUrl(url)
        print("HANDLE URL SUCCESS FROM UNIVERSAL LINK")
      } catch {
        print("HANDLE URL FAILURE FROM UNIVERSAL LINK")
        print(error)
        print(error.localizedDescription)
        print(Mirror(reflecting: error))
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
