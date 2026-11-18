(() => {
  'use strict';
  const app = document.getElementById('pg-app');
  if (!app) return;
  const $ = (selector) => app.querySelector(selector);
  const $$ = (selector) => Array.from(app.querySelectorAll(selector));
  const svgNS = 'http://www.w3.org/2000/svg';
  const canvas = $('#pg-canvas');
  const overlay = $('#pg-overlay');
  const image = $('#pg-image');
  const loading = $('#pg-loading');
  const status = $('#pg-status');
  const inspection = $('#pg-inspection');
  const modeButtons = $$('[data-pg-mode]');
  const overlayInputs = $$('[data-pg-overlay]');
  const cache = new Map();
  const state = { manifest: null, scene: null, data: null, mode: 'rgb', selected: null, loadToken: 0, imageToken: 0,
    opacity: 45, overlays: {masks: false, boundaries: false, boxes: false, ids: false, quality: false, centers: false, pose: false} };
  const modes = {
    rgb: {label:'RGB', title:'The camera observation', text:'The recorded color frame is the starting point. It is paired with an aligned depth image and camera intrinsics.', overlays:{}},
    depth: {label:'DEPTH', title:'Depth and scene geometry', text:'The registered depth frame measures distance from the camera. Depth and a fitted support plane help distinguish the exposed top layer from lower bricks.', overlays:{}},
    segmentation: {label:'SAM 3', title:'Candidate segmentation', text:'The separately recorded RGB-only example shows actual saved SAM 3 instance masks. These are candidates from an earlier standalone run, not the RGB-D top-layer detections.', overlays:{masks:true, boundaries:true}},
    detection: {label:'DETECTION', title:'Top-layer brick detection', text:'The pipeline combines top-plane height evidence with LEGO lattice and geometry priors to recover and validate oriented brick proposals.', overlays:{boxes:true, ids:true, centers:true}},
    quality: {label:'QUALITY', title:'Color-based quality classification', text:'Each detected brick is measured using corrected Lab color and compared with reference prototypes. Green G and red B indicate predicted quality, not ground truth.', overlays:{boxes:true, ids:true, quality:true}},
    pose: {label:'3D POSE', title:'Camera-frame geometry', text:'Camera intrinsics project the saved 3D cuboid corners onto the image. The exported center, orientation and corners are in the camera optical frame.', overlays:{boxes:true, ids:true, centers:true, pose:true}}
  };
  const MODE_ORDER = ['rgb', 'depth', 'segmentation', 'detection', 'quality', 'pose'];
  const getJSON = async (path) => {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Unable to load ${path} (HTTP ${response.status})`);
    return response.json();
  };
  const element = (name, attrs = {}) => {
    const node = document.createElementNS(svgNS, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  };
  const text = (value) => value === null || value === undefined || value === '' ? 'Unavailable' : String(value);
  const formatPoint = (value, places) => Array.isArray(value) && value.length >= 2 && value.every(Number.isFinite)
    ? `[${value.map((n) => n.toFixed(places)).join(', ')}]` : 'Unavailable';
  function setStatus(message, error = true) {
    status.textContent = message;
    status.classList.toggle('pg-status-error', error);
  }
  function showInspection() {
    inspection.replaceChildren();
    const label = document.createElement('span'); label.className = 'pg-overline'; label.textContent = 'BRICK INSPECTION';
    const title = document.createElement('h3');
    const brick = state.data && state.data.bricks.find((entry) => String(entry.id) === String(state.selected));
    if (!brick) {
      title.textContent = 'No brick selected';
      const hint = document.createElement('p'); hint.textContent = 'Select a brick in the viewer to inspect its saved quality and camera-frame geometry.';
      inspection.append(label, title, hint); $('#pg-clear').hidden = true; return;
    }
    const isMaskScene = !!brick.mask_path;
    title.textContent = `${isMaskScene ? 'SAM 3 candidate' : 'Brick'} #${brick.id}`;
    const description = document.createElement('p');
    description.textContent = isMaskScene ? 'Saved candidate mask and metadata from the separate RGB-only run.' : 'Saved prediction and geometry for this recorded frame.';
    const fields = [
      ['Predicted quality', brick.quality ? text(brick.quality).toUpperCase() : 'Unavailable for this RGB-only segmentation scene'],
      [isMaskScene ? 'SAM candidate color' : 'Predicted color', text(brick.colour).replaceAll('_', ' ')],
      [isMaskScene ? 'Image centroid [u, v] px' : 'Image center [u, v] px', formatPoint(brick.center, 1)],
      ['Top-face center [x, y, z] m', formatPoint(brick.top_face_center_camera_m, 4)],
      ['Image-plane orientation', Number.isFinite(brick.angle_deg) ? `${brick.angle_deg.toFixed(2)}°` : 'Unavailable'],
      ['Detector source', isMaskScene ? 'Standalone SAM 3 RGB-only run' : text(brick.source)],
      ['Color distance', Number.isFinite(brick.colour_distance) ? `${brick.colour_distance.toFixed(3)} (weighted prototype distance)` : 'Unavailable'],
      ['Confidence category', brick.confidence_category ? `${brick.confidence_category} (heuristic, not a probability)` : 'Unavailable'],
      ['Segmentation area', Number.isInteger(brick.mask_area) ? `${brick.mask_area} pixels` : 'Unavailable: instance masks were not saved'],
      ...(Number.isFinite(brick.sam_score) ? [['SAM 3 candidate score', `${brick.sam_score.toFixed(3)} (saved model score)`]] : [])
    ];
    const dl = document.createElement('dl');
    fields.forEach(([name, value]) => {
      const div = document.createElement('div'), dt = document.createElement('dt'), dd = document.createElement('dd');
      dt.textContent = name; dd.textContent = value; div.append(dt, dd); dl.append(div);
    });
    const geometry = document.createElement('details'); geometry.className = 'pg-geometry';
    const summary = document.createElement('summary'); summary.textContent = 'Camera-frame orientation and 3D corners';
    const geometryText = document.createElement('pre');
    const matrix = Array.isArray(brick.rotation_matrix_camera) && brick.rotation_matrix_camera.length === 3
      ? brick.rotation_matrix_camera.map((row) => formatPoint(row, 4)).join('\n') : 'Unavailable';
    const corners = Array.isArray(brick.box_corners_camera_m) && brick.box_corners_camera_m.length === 8
      ? brick.box_corners_camera_m.map((point, index) => `${index < 4 ? 'Top' : 'Bottom'} ${index % 4 + 1}: ${formatPoint(point, 4)} m`).join('\n') : 'Unavailable';
    geometryText.textContent = `Rotation matrix (camera frame):\n${matrix}\n\nCuboid corners (camera frame):\n${corners}`;
    geometry.append(summary, geometryText);
    inspection.append(label, title, description, dl, geometry); $('#pg-clear').hidden = false;
  }
  function selectBrick(id, refocus = false) {
    state.selected = String(state.selected) === String(id) ? null : id;
    renderOverlay(); showInspection();
    if (refocus) {
      const target = Array.from(overlay.querySelectorAll('[data-brick-id]')).find((node) => node.dataset.brickId === String(id));
      if (target) target.focus({preventScroll:true});
    }
  }
  function renderOverlay() {
    overlay.replaceChildren();
    if (!state.data) return;
    overlay.setAttribute('viewBox', `0 0 ${state.data.width} ${state.data.height}`);
    const enabled = state.overlays;
    state.data.bricks.forEach((brick) => {
      const isSelected = String(state.selected) === String(brick.id);
      const isMask = !!brick.mask_path;
      const descriptor = isMask ? `SAM 3 candidate ${brick.id}` : `brick ${brick.id}, predicted ${brick.quality || 'unknown'}`;
      const group = element('g', {class:`pg-brick${state.selected !== null && !isSelected ? ' dimmed' : ''}`, tabindex:'0', role:'button',
        'aria-label':`Inspect ${descriptor}`, 'aria-pressed':isSelected, 'data-brick-id':brick.id});
      const quality = brick.quality === 'good' || brick.quality === 'bad' ? brick.quality : 'unknown';
      const polygon = Array.isArray(brick.corners) ? brick.corners.map((point) => point.join(',')).join(' ') : null;
      if (isMask && enabled.masks) group.append(element('image', {href:brick.mask_image,x:0,y:0,width:state.data.width,height:state.data.height,opacity:state.opacity/100,class:'pg-mask-image'}));
      if (enabled.pose && Array.isArray(brick.projected_box_corners) && brick.projected_box_corners.length === 8) {
        const c = brick.projected_box_corners;
        [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]].forEach(([a,b]) =>
          group.append(element('line', {x1:c[a][0],y1:c[a][1],x2:c[b][0],y2:c[b][1],class:'pg-wire'})));
      }
      if (isMask) {
        group.append(element('path', {d:brick.mask_path,'fill-rule':'evenodd',class:`pg-outline pg-boundary${isSelected ? ' selected' : ''}${!enabled.boundaries && !isSelected ? ' quiet' : ''}`}));
      } else if (polygon) {
        group.append(element('polygon', {points:polygon, class:`pg-outline ${enabled.quality ? quality : ''}${isSelected ? ' selected' : ''}${!enabled.boxes && !isSelected ? ' quiet' : ''}`}));
      }
      if (enabled.centers && Array.isArray(brick.center)) {
        const [x,y] = brick.center;
        group.append(element('line', {x1:x-3,y1:y,x2:x+3,y2:y,class:'pg-center'}),
                     element('line', {x1:x,y1:y-3,x2:x,y2:y+3,class:'pg-center'}));
      }
      if ((enabled.ids || enabled.quality) && Array.isArray(brick.center)) {
        const parts = [];
        if (enabled.ids) parts.push(`#${brick.id}`);
        if (enabled.quality) parts.push(quality === 'good' ? 'G' : quality === 'bad' ? 'B' : '?');
        const label = element('text', {x:brick.center[0], y:brick.center[1]+4, 'text-anchor':'middle', class:`pg-label ${quality}`});
        label.textContent = parts.join(' '); group.append(label);
      }
      group.append(isMask ? element('path', {d:brick.mask_path,'fill-rule':'evenodd',class:'pg-hit'}) : element('polygon', {points:polygon,class:'pg-hit'}));
      group.addEventListener('click', (event) => { event.stopPropagation(); selectBrick(brick.id); });
      group.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectBrick(brick.id, true); }
      });
      overlay.append(group);
    });
  }
  function updateExplanation() {
    const mode = modes[state.mode];
    $('#pg-current-mode').textContent = mode.label;
    $('#pg-explanation-label').textContent = `${String(MODE_ORDER.indexOf(state.mode)+1).padStart(2,'0')} · ${mode.label}`;
    $('#pg-explanation-title').textContent = mode.title;
    $('#pg-explanation-text').textContent = mode.text;
    modeButtons.forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.pgMode === state.mode)));
    overlayInputs.forEach((input) => { input.checked = !!state.overlays[input.dataset.pgOverlay]; });
    $('#pg-depth-scale').hidden = state.mode !== 'depth';
    $('#pg-legend').hidden = !state.overlays.quality;
  }
  function showImage() {
    if (!state.scene || !state.data) return;
    const token = ++state.imageToken;
    const isDepth = state.mode === 'depth';
    const source = isDepth ? state.scene.depth : state.scene.rgb;
    loading.textContent = 'Loading image…'; loading.hidden = false;
    const next = new Image();
    next.onload = () => {
      if (token !== state.imageToken) return;
      image.src = source;
      image.alt = isDepth ? `Registered depth visualization for ${state.scene.name}` : `Original RGB frame for ${state.scene.name}`;
      loading.hidden = true;
    };
    next.onerror = () => {
      if (token !== state.imageToken) return;
      loading.textContent = 'Image could not be loaded'; setStatus(`Unable to load image: ${source}`);
    };
    next.src = source;
    if (state.data.depth) {
      $('#pg-depth-min').textContent = state.data.depth.display_min_m.toFixed(3);
      $('#pg-depth-max').textContent = state.data.depth.display_max_m.toFixed(3);
    }
  }
  function setMode(mode) {
    if (!state.scene || !state.data || !modes[mode] || !state.scene.available[mode]) return;
    const wasDepth = state.mode === 'depth';
    state.mode = mode;
    if (mode === 'rgb') { state.selected = null; showInspection(); }
    Object.keys(state.overlays).forEach((key) => { state.overlays[key] = !!modes[mode].overlays[key]; });
    updateExplanation(); renderOverlay();
    if (wasDepth !== (mode === 'depth')) showImage();
  }
  function updateAvailability() {
    const available = state.scene.available;
    modeButtons.forEach((button) => {
      const permitted = !!available[button.dataset.pgMode];
      button.disabled = !permitted;
      let tag = button.querySelector('small');
      if (!tag) { tag = document.createElement('small'); tag.textContent = 'Unavailable'; button.append(tag); }
      tag.hidden = permitted;
      button.title = permitted ? '' : (button.dataset.pgMode === 'segmentation' ? state.data.masks_reason : 'Not saved for this scene');
    });
    const permissions = {masks:available.segmentation, boundaries:available.segmentation, boxes:available.detection,
                         ids:true, quality:available.quality, centers:true, pose:available.pose};
    overlayInputs.forEach((input) => {
      const permitted = !!permissions[input.dataset.pgOverlay];
      input.disabled = !permitted;
      input.closest('label').classList.toggle('pg-disabled', !permitted);
      const tag = input.closest('label').querySelector('small');
      if (tag) tag.hidden = permitted;
      if (!permitted) state.overlays[input.dataset.pgOverlay] = false;
    });
    const slider = $('#pg-opacity');
    slider.disabled = !available.segmentation;
    slider.closest('label').classList.toggle('pg-disabled', slider.disabled);
    $('#pg-opacity-value').textContent = `${state.opacity}%${slider.disabled ? ' · unavailable' : ''}`;
    $('#pg-missing').textContent = available.segmentation
      ? 'This is a separate earlier RGB-only SAM 3 run. Depth, top-layer detection, quality, and pose outputs were not saved for this frame.'
      : 'The top-layer RGB-D recordings do not contain saved SAM 3 instance masks. Select the separate RGB-only scene to explore genuine masks.';
  }
  async function selectScene(scene) {
    const token = ++state.loadToken;
    state.imageToken++;
    state.selected = null; state.scene = null; state.data = null;
    overlay.replaceChildren(); showInspection();
    loading.textContent = 'Loading recorded scene…'; loading.hidden = false; setStatus('');
    modeButtons.forEach((button) => { button.disabled = true; });
    $$('.pg-scene').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.sceneId === scene.id)));
    try {
      const data = cache.get(scene.id) || await getJSON(scene.metadata);
      if (token !== state.loadToken) return;
      if (data.id !== scene.id || !Array.isArray(data.bricks) || !data.width || !data.height) throw new Error('Scene metadata does not match its manifest');
      cache.set(scene.id, data); state.scene = scene; state.data = data;
      if (!scene.available[state.mode]) { state.mode = 'rgb'; Object.keys(state.overlays).forEach((key) => { state.overlays[key] = false; }); }
      canvas.style.aspectRatio = `${data.width} / ${data.height}`;
      $('#pg-scene-title').textContent = scene.name;
      $('#pg-frame-caption').textContent = data.frame === null
        ? `Separate RGB-only SAM 3 run · ${data.bricks.length} saved masks`
        : `${data.dataset} · ${data.recording} · frame ${data.frame} · ${data.bricks.length} saved candidates`;
      updateAvailability(); updateExplanation(); showInspection(); renderOverlay(); showImage();
    } catch (error) {
      if (token !== state.loadToken) return;
      loading.textContent = 'Scene could not be loaded'; setStatus(error.message || String(error));
    }
  }
  async function initialize() {
    try {
      const manifest = await getJSON('playground/manifest.json');
      if (!Array.isArray(manifest.scenes) || !manifest.scenes.length) throw new Error('No recorded scenes are available');
      state.manifest = manifest;
      const container = $('#pg-scenes'); container.replaceChildren();
      manifest.scenes.forEach((scene) => {
        const button = document.createElement('button'); button.type = 'button'; button.className = 'pg-scene';
        button.dataset.sceneId = scene.id; button.setAttribute('aria-pressed','false');
        const thumb = document.createElement('img'); thumb.src = scene.thumbnail; thumb.alt = ''; thumb.loading = 'lazy';
        const copy = document.createElement('span'), title = document.createElement('strong'), desc = document.createElement('small');
        title.textContent = scene.name; desc.textContent = scene.description; copy.append(title, desc); button.append(thumb, copy);
        button.addEventListener('click', () => selectScene(scene)); container.append(button);
      });
      await selectScene(manifest.scenes[0]);
    } catch (error) {
      $('#pg-scenes').textContent = 'Recorded scenes could not be loaded.';
      loading.textContent = 'Playground unavailable'; setStatus(error.message || String(error));
      modeButtons.forEach((button) => { button.disabled = true; });
    }
  }
  modeButtons.forEach((button) => button.addEventListener('click', () => setMode(button.dataset.pgMode)));
  overlayInputs.forEach((input) => input.addEventListener('change', () => {
    state.overlays[input.dataset.pgOverlay] = input.checked;
    updateExplanation(); renderOverlay();
  }));
  $('#pg-opacity').addEventListener('input', (event) => {
    state.opacity = Number(event.target.value);
    $('#pg-opacity-value').textContent = `${state.opacity}%`;
    renderOverlay();
  });
  overlay.addEventListener('click', (event) => {
    if (event.target === overlay) { state.selected = null; renderOverlay(); showInspection(); }
  });
  $('#pg-clear').addEventListener('click', () => { state.selected = null; renderOverlay(); showInspection(); });
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) { observer.disconnect(); initialize(); }
    }, {rootMargin:'400px'});
    observer.observe(app);
  } else initialize();
})();
