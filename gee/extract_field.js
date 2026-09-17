/*
  GEE Code Editor Script — Extract field features for local inference.

  HOW TO USE:
  ─────────────────────────────────────────────────────────────────────
  1. Open https://code.earthengine.google.com/
  2. Paste this entire script into the Code Editor
  3. Click ▶ Run
  4. Use the drawing tools (top-left of map) to draw a polygon over a field
  5. Select the season ('rabi' or 'kharif') in the dropdown on the left panel
  6. Click the "Extract Features" button
  7. JSON will be printed in the Console — copy the JSON block
  8. Paste it into  inference/predict_from_gee.py  locally to get the prediction
  ─────────────────────────────────────────────────────────────────────

  IMPORTANT: The processing (bands, indices, cloud mask, reducer) is
  identical to pipeline/gee_extract.py so the model gets the same
  feature distribution it was trained on.
*/

// ================================================================
//  Configuration
// ================================================================

var S2_COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED';

var SPECTRAL_BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];

// SCL clear-land classes (same as training)
var SCL_CLEAR = [4, 5, 6, 7];

// NOTE: These windows MUST match the training years so the input distribution
// matches what the model + StandardScaler were fit on:
//   Rabi   = Oct 2022 -> Apr 2023,  Kharif = May 2023 -> Nov 2023.
// To predict a field in a different year, change the year here AND retrain /
// re-fit the scaler on that year (otherwise reflectance offsets bias results).
var SEASON_WINDOWS = {
  'rabi':   {start: '2022-10-01', end: '2023-04-30',
             months: ['2022-10','2022-11','2022-12','2023-01','2023-02','2023-03','2023-04']},
  'kharif': {start: '2023-05-01', end: '2023-11-30',
             months: ['2023-05','2023-06','2023-07','2023-08','2023-09','2023-10','2023-11']}
};

// Feature names in exact training order (28 features)
var FEATURE_NAMES = [
  'B2_mean','B3_mean','B4_mean','B5_mean','B6_mean','B7_mean','B8_mean','B8A_mean',
  'B11_mean','B12_mean','NDVI_mean','EVI_mean','NDWI_mean','SAVI_mean',
  'B2_stdDev','B3_stdDev','B4_stdDev','B5_stdDev','B6_stdDev','B7_stdDev',
  'B8_stdDev','B8A_stdDev','B11_stdDev','B12_stdDev','NDVI_stdDev','EVI_stdDev',
  'NDWI_stdDev','SAVI_stdDev'
];

// ================================================================
//  Cloud masking (identical to training)
// ================================================================

function maskS2Clouds(image) {
  var scl = image.select('SCL');
  var clear = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(7));
  return image.updateMask(clear);
}

// ================================================================
//  Vegetation indices (identical to training)
// ================================================================

function addIndices(image) {
  var ndvi = image.normalizedDifference(['B8','B4']).rename('NDVI');

  var evi = image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))',
    {NIR: image.select('B8'), RED: image.select('B4'), BLUE: image.select('B2')}
  ).rename('EVI');

  var ndwi = image.normalizedDifference(['B8','B11']).rename('NDWI');

  var savi = image.expression(
    '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
    {NIR: image.select('B8'), RED: image.select('B4')}
  ).rename('SAVI');

  return image.addBands([ndvi, evi, ndwi, savi]);
}

// ================================================================
//  Monthly composite (identical to training)
// ================================================================

function monthlyComposite(startDate, endDate, bounds) {
  return ee.ImageCollection(S2_COLLECTION)
    .filterDate(startDate, endDate)
    .filterBounds(bounds)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
    .map(maskS2Clouds)
    .select(SPECTRAL_BANDS)
    .map(addIndices)
    .median();
}

// ================================================================
//  Extract features for a drawn polygon
//  OPTIMISED: stacks all 7 months into one image → single reduceRegion
// ================================================================

function extractFeatures(geometry, season) {
  var config = SEASON_WINDOWS[season];
  if (!config) {
    print('ERROR: Invalid season "' + season + '". Use "rabi" or "kharif".');
    return;
  }

  var months = config.months;
  var allBands = SPECTRAL_BANDS.concat(['NDVI','EVI','NDWI','SAVI']);

  // Combined mean+stdDev reducer (same as training)
  var reducer = ee.Reducer.mean().combine({
    reducer2: ee.Reducer.stdDev(),
    sharedInputs: true
  });

  // ── Build one stacked image: 7 months × 14 bands = 98 bands ──
  // After reduceRegion with mean+stdDev → 196 values (7 × 28)
  var stacked = ee.Image();
  var stackedBandNames = [];

  for (var m = 0; m < months.length; m++) {
    var parts = months[m].split('-');
    var year  = parseInt(parts[0]);
    var month = parseInt(parts[1]);
    var startDate = ee.Date.fromYMD(year, month, 1);
    var endDate   = startDate.advance(1, 'month');

    var composite = monthlyComposite(startDate, endDate, geometry);

    // Rename bands to t0_B2, t0_B3, ... t0_SAVI, t1_B2, ...
    var prefix = 't' + m + '_';
    var renamed = composite.select(allBands).rename(
      allBands.map(function(b) { return prefix + b; })
    );

    stacked = stacked.addBands(renamed);

    // Track the band names for the reducer output
    for (var b = 0; b < allBands.length; b++) {
      stackedBandNames.push(prefix + allBands[b]);
    }
  }

  // Remove the empty initial band if present
  stacked = stacked.select(stackedBandNames);

  // ── Single reduceRegion call (not reduceRegions) ──
  var stats = stacked.reduceRegion({
    reducer: reducer,
    geometry: geometry,
    scale: 10,
    maxPixels: 1e7
  });

  // ── One evaluate() call for all 196 values ──
  stats.evaluate(function(result, error) {
    if (error) {
      print('ERROR: ' + error);
      return;
    }

    // Parse the flat dict into (7, 28) array
    // GEE names: t0_B2_mean, t0_B2_stdDev, ... t6_SAVI_stdDev
    var features = [];

    for (var t = 0; t < months.length; t++) {
      var prefix = 't' + t + '_';
      var timestepValues = [];

      for (var f = 0; f < FEATURE_NAMES.length; f++) {
        // FEATURE_NAMES[f] = "B2_mean" → GEE key = "t0_B2_mean"
        var key = prefix + FEATURE_NAMES[f];
        var val = result[key];
        if (val === null || val === undefined) {
          val = 0.0;
        }
        timestepValues.push(val);
      }
      features.push(timestepValues);
    }

    // Output JSON for pasting into Python
    var output = {
      season: season,
      shape: [1, 7, 28],
      features: features,
      months: months,
      note: 'Paste this JSON into inference/predict_from_gee.py'
    };

    print('════════════════════════════════════════════════════════════');
    print('  FEATURE EXTRACTION COMPLETE — ' + season.toUpperCase());
    print('  Copy the JSON below and paste into predict_from_gee.py');
    print('════════════════════════════════════════════════════════════');
    print(JSON.stringify(output));

    // Human-readable summary
    print('');
    print('Human-readable features per month:');
    for (var m2 = 0; m2 < months.length; m2++) {
      var p = 't' + m2 + '_';
      var ndvi = result[p + 'NDVI_mean'];
      var evi  = result[p + 'EVI_mean'];
      var b8   = result[p + 'B8_mean'];
      print(months[m2] +
        ': NDVI=' + (ndvi !== null ? ndvi.toFixed(4) : 'N/A') +
        '  EVI='  + (evi  !== null ? evi.toFixed(4)  : 'N/A') +
        '  B8='   + (b8   !== null ? b8.toFixed(1)   : 'N/A'));
    }
  });
}

// ================================================================
//  UI — Drawing tools + season selector + button
// ================================================================

// Create drawing tools
var drawingTools = Map.drawingTools();
drawingTools.setShown(true);
drawingTools.setLinked(true);

// Season selector
var seasonSelect = ui.Select({
  items: ['rabi', 'kharif'],
  value: 'rabi',
  style: {width: '200px'}
});

// Extract button
var extractButton = ui.Button({
  label: '🌾 Extract Features',
  style: {stretch: 'horizontal', color: 'green'},
  onClick: function() {
    var layers = drawingTools.layers();
    if (layers.length() === 0) {
      print('ERROR: Draw a polygon on the map first!');
      return;
    }

    // Get the last drawn geometry
    var lastLayer = layers.get(layers.length() - 1);
    var geometry = lastLayer.toGeometry();
    var season = seasonSelect.getValue();

    print('Extracting ' + season + ' features for drawn polygon...');
    print('This may take 30-60 seconds...');

    extractFeatures(geometry, season);
  }
});

// Clear button
var clearButton = ui.Button({
  label: '🗑️ Clear Drawings',
  onClick: function() {
    drawingTools.layers().reset();
  }
});

// Panel
var panel = ui.Panel({
  widgets: [
    ui.Label('Crop Type Inference — Feature Extractor', {fontWeight: 'bold', fontSize: '16px'}),
    ui.Label('1. Draw a polygon over a field on the map'),
    ui.Label('2. Select season:'),
    seasonSelect,
    ui.Label('3. Click Extract:'),
    extractButton,
    clearButton,
    ui.Label('─────────────────────────────'),
    ui.Label('4. Copy JSON from Console'),
    ui.Label('5. Paste into predict_from_gee.py'),
  ],
  style: {width: '300px', padding: '10px'}
});

ui.root.insert(0, panel);

// Center map on Pakistan Punjab
Map.setCenter(72.0, 30.5, 7);
Map.setOptions('HYBRID');
