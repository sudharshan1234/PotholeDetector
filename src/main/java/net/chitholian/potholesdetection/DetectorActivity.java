/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package net.chitholian.potholesdetection;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.location.Address;
import android.location.Geocoder;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.Looper;
import android.util.Log;
import android.util.Size;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;

import net.chitholian.potholesdetection.OverlayView.DrawCallback;
import net.chitholian.potholesdetection.env.ImageUtils;
import net.chitholian.potholesdetection.env.Logger;
import net.chitholian.potholesdetection.tracking.MultiBoxTracker;

import android.media.MediaPlayer;

import androidx.core.app.ActivityCompat;

import com.google.android.gms.common.internal.Objects;
import com.google.android.gms.location.FusedLocationProviderClient;
import com.google.android.gms.location.LocationCallback;
import com.google.android.gms.location.LocationRequest;
import com.google.android.gms.location.LocationResult;
import com.google.android.gms.location.LocationServices;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.common.ConnectionResult;
import com.google.android.gms.common.api.GoogleApiClient;
import com.google.android.gms.location.LocationServices;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener{
  private static final Logger LOGGER = new Logger();
  protected Context context;
  private static final int REQUEST_CODE_LOCATION_PERMISSION =1;
  double lat;
  double lon;
  String provider;
  private MediaPlayer mp;
  int k=0;
  int i=0;
  StringBuilder strReturnedAddress = new StringBuilder("");
  String address;
  String city;
  String state;
  String country;
  String postalCode;

  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_MODEL_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "potholes.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/potholes.txt";

  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;



  private static final boolean MAINTAIN_ASPECT = false;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(480, 480);

  private Integer sensorOrientation;

  private Classifier detector;
  private FirebaseDatabase mDatabase;
  private DatabaseReference mDatabaseReference;

  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private byte[] luminanceCopy;


  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector = TFLiteObjectDetectionAPIModel.create(
              getAssets(), TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE, TF_OD_API_MODEL_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
              Toast.makeText(
                      getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
            ImageUtils.getTransformationMatrix(
                    previewWidth, previewHeight,
                    cropSize, cropSize,
                    sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
            new DrawCallback() {
              @Override
              public void drawCallback(final Canvas canvas) {
                tracker.draw(canvas);
                if (isDebug()) {
                  tracker.drawDebug(canvas);
                }
              }
            });

    addCallback(
            new DrawCallback() {
              @Override
              public void drawCallback(final Canvas canvas) {
                if (!isDebug()) {
                  return;
                }
                final Bitmap copy = cropCopyBitmap;
                if (copy == null) {
                  return;
                }

                final int backgroundColor = Color.argb(100, 0, 0, 0);
                canvas.drawColor(backgroundColor);

                final Matrix matrix = new Matrix();
                final float scaleFactor = 2;
                matrix.postScale(scaleFactor, scaleFactor);
                matrix.postTranslate(
                        canvas.getWidth() - copy.getWidth() * scaleFactor,
                        canvas.getHeight() - copy.getHeight() * scaleFactor);
                canvas.drawBitmap(copy, matrix, new Paint());
              }
            });
  }

  OverlayView trackingOverlay;

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    byte[] originalLuminance = getLuminance();
    tracker.onFrame(
            previewWidth,
            previewHeight,
            getLuminanceStride(),
            sensorOrientation,
            originalLuminance,
            timestamp);
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    if (luminanceCopy == null) {
      luminanceCopy = new byte[originalLuminance.length];
    }
    System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    runInBackground(
            new Runnable() {
              @Override
              public void run() {
                LOGGER.i("Running detection on image " + currTimestamp);
                final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);

                cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                final Canvas canvas = new Canvas(cropCopyBitmap);
                final Paint paint = new Paint();
                paint.setColor(Color.RED);
                paint.setStyle(Style.STROKE);
                paint.setStrokeWidth(2.0f);

                final List<Classifier.Recognition> mappedRecognitions =
                        new LinkedList<Classifier.Recognition>();

                for (final Classifier.Recognition result : results) {
                  final RectF location = result.getLocation();
                  if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {
                    canvas.drawRect(location, paint);

                    cropToFrameTransform.mapRect(location);
                    result.setLocation(location);
                    mappedRecognitions.add(result);
                  }

                }

                tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);

                if (tracker.trackedObjects.size() == 0) {
                  k=0;
                  if (mp.isPlaying()) {
                    mp.pause();
                    mp.seekTo(0);
                  }
                } else {
                  if(k==0)
                  {
                    getCurrentLocation();
                    if((lat!=0)&&(lon!=0)) {
                      String mGroupId = mDatabase.getReference().push().getKey();
                      mDatabaseReference = mDatabase.getReference().child("PotholeLocation").child(mGroupId).child("LatLon");
                      mDatabaseReference.setValue(lat+","+lon);
                      mDatabaseReference = mDatabase.getReference().child("PotholeLocation").child(mGroupId).child("Address");
                      mDatabaseReference.setValue(address);
                      mDatabaseReference = mDatabase.getReference().child("PotholeLocation").child(mGroupId).child("City");
                      mDatabaseReference.setValue(city);
                      mDatabaseReference = mDatabase.getReference().child("PotholeLocation").child(mGroupId).child("State");
                      mDatabaseReference.setValue(state);
                      mDatabaseReference = mDatabase.getReference().child("PotholeLocation").child(mGroupId).child("PostalCode");
                      mDatabaseReference.setValue(postalCode);

                    }
                    i++;
                  }
                  if (!mp.isPlaying()) {
                    mp.start();
                  }


                }
                trackingOverlay.postInvalidate();

                requestRender();
                computingDetection = false;
              }
            });
  }



  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  public void onSetDebug(final boolean debug) {
    detector.enableStatLogging(debug);
  }


  @SuppressLint("MissingPermission")
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    mp = MediaPlayer.create(this, R.raw.alarm);

    mDatabase = FirebaseDatabase.getInstance();
    mp.setLooping(true);
  }

  private void getCurrentLocation()
  {
    final LocationRequest locationRequest=new LocationRequest();
    k=1;
    locationRequest.setInterval(10000);
    locationRequest.setFastestInterval(3000);
    locationRequest.setPriority(LocationRequest.PRIORITY_HIGH_ACCURACY);
    LocationServices.getFusedLocationProviderClient(DetectorActivity.this)
            .requestLocationUpdates(locationRequest, new LocationCallback(){
              @Override
              public void onLocationResult(LocationResult locationResult) {
                super.onLocationResult(locationResult);
                LocationServices.getFusedLocationProviderClient(DetectorActivity.this)
                        .removeLocationUpdates(this);
                if(locationResult!=null && locationResult.getLocations().size()>0){
                  int latestLocationIndex = locationResult.getLocations().size()-1;
                  lat=locationResult.getLocations().get(latestLocationIndex).getLatitude();
                  lon=locationResult.getLocations().get(latestLocationIndex).getLongitude();
                  Log.d("ADebugTag", "Latitude: " + Double.toString(lat));
                  Log.d("ADebugTag", "Longitude: " + Double.toString(lon));
                  getCompleteAddressString(lat,lon);
                }
              }
            }, Looper.getMainLooper());

  }
  private String getCompleteAddressString(double LATITUDE, double LONGITUDE) {
    String strAdd = "";
    Geocoder geocoder = new Geocoder(this, Locale.getDefault());
    try {
      List<Address> addresses = geocoder.getFromLocation(LATITUDE, LONGITUDE, 1);
      if (addresses != null) {
        Address returnedAddress = addresses.get(0);
        address = returnedAddress.getAddressLine(0); // If any additional address line present than only, check with max available address lines by getMaxAddressLineIndex()
        city = returnedAddress.getLocality();
        state = returnedAddress.getAdminArea();
        country = returnedAddress.getCountryName();
        postalCode = returnedAddress.getPostalCode();

        for (int i = 0; i <= returnedAddress.getMaxAddressLineIndex(); i++) {
          strReturnedAddress.append(returnedAddress.getAddressLine(i));
        }
        strAdd = strReturnedAddress.toString();
        Log.w("Current location: ", strReturnedAddress.toString());
      } else {
        Log.w("Current location: ", "No Address returned!");
      }
    } catch (Exception e) {
      e.printStackTrace();
      Log.w("Current location", "Cannot get Address!");
    }
    return strAdd;
  }


  @Override
  public synchronized void onResume() {
    if (mp.isPlaying()) {
      mp.pause();
      mp.seekTo(0);
    }
    super.onResume();
  }

  @Override
  public synchronized void onStop() {

    super.onStop();
  }

  @Override
  public synchronized void onDestroy() {
    mp.release();
    super.onDestroy();
  }
}
