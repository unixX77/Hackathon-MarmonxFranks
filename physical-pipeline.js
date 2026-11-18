(() => {
  const explorer = document.getElementById('physical-explorer');
  if (!explorer) return;
  const cards = Array.from(explorer.querySelectorAll('.explorer-card'));
  const panel = document.getElementById('physical-stage-detail');
  const content = panel.querySelector('.panel-content');
  const close = panel.querySelector('.detail-close');
  const announcement = document.getElementById('physical-stage-announcement');
  const fields = Object.fromEntries(['number', 'title', 'summary', 'how', 'input', 'process', 'output', 'caption', 'image'].map((key) => [key, document.getElementById(`physical-detail-${key}`)]));
  const stages = [
    { number:'STAGE 01 · CAPTURE & SEGMENT', title:'Capture and isolate the brick', summary:'A brick mask defines the region to inspect.', how:'An RGB camera captures the scene. SAM 3 segments individual bricks, producing a mask for each classifier input.', input:'RGB image of one or more bricks', process:'SAM 3 segmentation and mask selection', output:'Individual brick masks and image regions', image:'assets/physical-brick-segmented.jpg', alt:'A real brick outlined by the saved classifier output', caption:'A saved SAM 3 segmentation and classifier output.' },
    { number:'STAGE 02 · COLOR & TEXTURE', title:'Approach 5 appearance features', summary:'Hand-designed features summarize the brick’s appearance.', how:'Inside the mask, the model measures LAB color distributions and grayscale texture statistics, excluding very dark and bright pixels.', input:'RGB image and an individual brick mask', process:'44 LAB histogram values plus 8 gradient, Laplacian and brightness statistics', output:'A 52-dimensional appearance feature vector', image:'assets/physical-brick-segmented.jpg', alt:'A segmented physical brick with a green outline', caption:'Appearance features are measured within the selected brick mask.' },
    { number:'STAGE 03 · DINOV2', title:'DINOv2 visual features', summary:'A frozen visual encoder describes the brick in image patches.', how:'The masked crop is passed through pretrained DINOv2-small. Patch-token mean, standard deviation and 90th percentile produce a 1,152-value representation.', input:'Masked brick crop with median-color background', process:'Frozen DINOv2 patch features; PCA reduces 1,152 values to 32', output:'A reduced learned visual representation', image:'assets/physical-brick-input.jpg', alt:'An example photograph of a physical brick', caption:'Example input image; DINOv2 receives a masked crop.' },
    { number:'STAGE 04 · CLASSIFY', title:'Two classifiers, one score', summary:'Approach 17 combines appearance and learned visual evidence.', how:'Each feature branch has a trained linear SVM. Their decision scores are averaged with equal weight to produce the combined score.', input:'Appearance features and DINOv2 representation', process:'Scaling, two linear SVMs and equal-weight late fusion', output:'A combined decision score and two component scores', image:'assets/physical-pipeline.svg', alt:'Diagram of the two classifier branches and their score fusion', caption:'The two feature branches contribute equally to the fused score.' },
    { number:'STAGE 05 · DECIDE', title:'Good or bad prediction', summary:'A binary threshold turns the fused score into a label.', how:'The selected policy labels a score at or below −2.0 good and a score above −2.0 bad. The live viewer displays each segmented brick with its prediction.', input:'Combined Approach 17 decision score', process:'Apply the −2 binary threshold', output:'Good or bad label and score for each detected brick', image:'assets/vision-poster.jpg', alt:'Live viewer showing physical bricks and their predictions', caption:'Live physical-brick inspection; scores are not probabilities.' }
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
    announcement.textContent = 'Physical-brick stage details closed';
    suppressFocus = true; cards[visible].focus(); suppressFocus = false;
  });
  explorer.addEventListener('keydown', (event) => { if (event.key === 'Escape' && !panel.hidden) close.click(); });
})();
