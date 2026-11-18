(() => {
  const explorer = document.getElementById('pipeline-explorer');
  if (!explorer) return;
  const cards = Array.from(explorer.querySelectorAll('.explorer-card'));
  const panel = document.getElementById('stage-detail');
  const content = panel.querySelector('.panel-content');
  const close = panel.querySelector('.detail-close');
  const announcement = document.getElementById('stage-announcement');
  const fields = Object.fromEntries(['number', 'title', 'summary', 'how', 'input', 'process', 'output', 'caption', 'image'].map((key) => [key, document.getElementById(`detail-${key}`)]));
  const stages = [
    { number:'STAGE 01 · CAPTURE', title:'RGB-D image acquisition', summary:'Aligned camera observations establish the scene geometry.', how:'The RGB-D camera provides aligned color and depth frames. Camera intrinsics describe how image pixels relate to camera rays.', input:'RGB image, depth map, camera intrinsics', process:'Capture and align color and depth', output:'Calibrated color and depth data', image:'assets/dense.png', alt:'Recorded LEGO stack shown with top-layer detection boxes', caption:'Recorded LEGO scene, shown with detection overlay.' },
    { number:'STAGE 02 · SEGMENT', title:'SAM 3 and depth-based segmentation', summary:'Appearance and geometric evidence work together.', how:'SAM 3 proposes LEGO-shaped masks. Temporal depth filtering and a fitted support plane provide geometric context for refining candidates.', input:'Aligned RGB-D observations', process:'SAM 3 masks, depth filtering, support-plane estimation and geometric filtering', output:'Candidate masks and scene geometry', image:'assets/dense.png', alt:'LEGO stack with top-layer detection overlay', caption:'Recorded scene with the resulting top-layer detections.' },
    { number:'STAGE 03 · DETECT', title:'Top-layer brick detection', summary:'The exposed plane and brick geometry identify accessible candidates.', how:'A supported height peak isolates the exposed top plane. LEGO size, orientation and lattice priors propose oriented boxes, including candidates SAM may miss or merge.', input:'Candidate masks, support-plane geometry and depth', process:'Top-plane isolation, lattice reconstruction and box validation', output:'Individual top-layer brick candidates and oriented boxes', image:'assets/single.png', alt:'A sparse LEGO stack with a single top-brick detection', caption:'Sparse-scene example with one remaining top brick.' },
    { number:'STAGE 04 · CLASSIFY & LOCATE', title:'Brick quality and 3D pose', summary:'Color classification and camera-frame localization run downstream.', how:'Corrected brick color is compared with reference prototypes for the LEGO good/bad label. Camera calibration and the fitted top plane yield the brick’s camera-frame geometry.', input:'Oriented boxes, RGB-D data and camera calibration', process:'Corrected Lab prototype comparison and projection onto the top plane', output:'Quality label, camera-frame center, orientation and estimated 3D corners', image:'assets/classified.png', alt:'LEGO stack with green and red quality classification boxes', caption:'LEGO color classification: green is good and red is bad.' },
    { number:'STAGE 05 · HANDOFF', title:'Perception to robotic manipulation', summary:'Brick observations are handed to the robotics team.', how:'The perception pipeline exports labels and camera-frame geometry. The robotics team handles camera-to-robot transformation, motion planning, grasping and sorting.', input:'Detected bricks, quality labels and camera-frame poses', process:'Camera-to-robot transformation, motion planning and grasp execution', output:'Targets for the robotics team’s pick-and-sort workflow', image:'assets/robot-poster.jpg', alt:'Franka robotic arm and gripper above a LEGO sorting setup', caption:'The robotics team’s manipulation setup.' }
  ];
  let pinned = 0;
  let visible = 0;
  let returnTimer = null;
  let suppressFocus = false;
  const cancelReturn = () => { if (returnTimer !== null) clearTimeout(returnTimer); returnTimer = null; };
  const show = (index) => {
    const stage = stages[index];
    if (!stage) return;
    const changed = visible !== index || panel.hidden;
    visible = index;
    panel.hidden = false;
    cards.forEach((card, i) => {
      card.classList.toggle('is-active', i === index);
      card.setAttribute('aria-pressed', String(i === pinned));
      const action = card.querySelector('.card-action');
      action.firstChild.nodeValue = i === pinned ? 'Pinned stage ' : 'Explore stage ';
      action.querySelector('span').textContent = i === pinned ? '✓' : '↗';
    });
    for (const key of ['number', 'title', 'summary', 'how', 'input', 'process', 'output', 'caption']) fields[key].textContent = stage[key];
    fields.image.src = stage.image;
    fields.image.alt = stage.alt;
    if (changed) {
      content.classList.remove('is-changing');
      void content.offsetWidth;
      content.classList.add('is-changing');
    }
  };
  const returnToPinned = () => {
    cancelReturn();
    returnTimer = setTimeout(() => {
      if (pinned === null) { panel.hidden = true; cards.forEach((card) => card.classList.remove('is-active')); }
      else show(pinned);
    }, 180);
  };
  cards.forEach((card, index) => {
    card.addEventListener('pointerenter', (event) => {
      if (event.pointerType === 'touch') return;
      cancelReturn(); show(index);
    });
    card.addEventListener('pointerleave', (event) => { if (event.pointerType !== 'touch') returnToPinned(); });
    card.addEventListener('focus', () => { if (!suppressFocus) { cancelReturn(); show(index); } });
    card.addEventListener('click', () => {
      cancelReturn(); pinned = index; show(index);
      announcement.textContent = `${stages[index].title} selected`;
    });
    card.addEventListener('keydown', (event) => {
      if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
        event.preventDefault();
        cards[(index + (event.key === 'ArrowRight' ? 1 : cards.length - 1)) % cards.length].focus();
      }
    });
  });
  panel.addEventListener('pointerenter', cancelReturn);
  panel.addEventListener('pointerleave', (event) => { if (event.pointerType !== 'touch') returnToPinned(); });
  close.addEventListener('click', () => {
    cancelReturn(); pinned = null; panel.hidden = true;
    cards.forEach((card) => {
      card.classList.remove('is-active'); card.setAttribute('aria-pressed', 'false');
      const action = card.querySelector('.card-action');
      action.firstChild.nodeValue = 'Explore stage '; action.querySelector('span').textContent = '↗';
    });
    announcement.textContent = 'Stage details closed';
    suppressFocus = true; cards[visible].focus(); suppressFocus = false;
  });
  explorer.addEventListener('keydown', (event) => { if (event.key === 'Escape' && !panel.hidden) close.click(); });
})();
