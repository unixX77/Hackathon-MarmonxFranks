(() => {
  const track = document.getElementById('pipeline-stages');
  if (!track) return;
  const cards = Array.from(track.querySelectorAll('.stage-card'));

  cards.forEach((card) => {
    card.addEventListener('click', () => {
      const open = card.getAttribute('aria-expanded') !== 'true';
      cards.forEach((other) => other.setAttribute('aria-expanded', 'false'));
      card.setAttribute('aria-expanded', String(open));
      if (open) card.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
    });
    card.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') card.setAttribute('aria-expanded', 'false');
    });
  });

  document.querySelectorAll('.stage-arrow').forEach((arrow) => {
    arrow.addEventListener('click', () => {
      track.scrollBy({ left: Number(arrow.dataset.scroll) * 260, behavior: 'smooth' });
    });
  });
})();
