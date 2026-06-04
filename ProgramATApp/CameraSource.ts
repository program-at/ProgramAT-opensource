/**
 * CameraSource
 *
 * Central definition of the camera sources that can feed the shared frame
 * pipeline. ToolRunner and the tools it runs are agnostic to the source — they
 * receive frames through CameraView's imperative handle regardless of whether
 * the frames originate from the phone camera or a pair of Meta Ray-Ban glasses.
 *
 * The active source is owned by CameraView (the single frame provider for
 * ToolRunner) and selected by the user via the camera-source buttons.
 */
export enum CameraSource {
  Phone = 'phone',
  RayBan = 'rayBan',
}
