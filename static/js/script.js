/* ─────────────────────────────────────────────────
   Fresh Meat Detector – script.js
   ───────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {

  const dropZone   = document.getElementById('drop-zone');
  const fileInput  = document.getElementById('file-input');
  const dzDefault  = document.getElementById('dz-default');
  const dzPreview  = document.getElementById('dz-preview');
  const previewImg = document.getElementById('preview-img');
  const removeBtn  = document.getElementById('remove-btn');
  const submitBtn  = document.getElementById('submit-btn');
  const btnText    = document.getElementById('btn-text');
  const btnLoading = document.getElementById('btn-loading');
  const form       = document.getElementById('upload-form');

  if (!dropZone) return; // guard: hanya jalan di halaman index

  // ── Drag & Drop ──────────────────────────────
  ['dragenter','dragover','dragleave','drop'].forEach(evt => {
    dropZone.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); });
  });

  dropZone.addEventListener('dragenter', () => dropZone.classList.add('dragover'));
  dropZone.addEventListener('dragover',  () => dropZone.classList.add('dragover'));
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

  dropZone.addEventListener('drop', e => {
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer?.files;
    if (files && files.length) handleFile(files[0]);
  });

  // ── Click to browse ──────────────────────────
  dropZone.addEventListener('click', e => {
    if (e.target === removeBtn || removeBtn.contains(e.target)) return;
    fileInput.click();
  });

  dropZone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  // ── Handle selected file ──────────────────────
  function handleFile(file) {
    if (!file.type.startsWith('image/')) {
      showToast('File tidak valid. Harap pilih gambar JPG atau PNG.', 'error');
      return;
    }

    const reader = new FileReader();
    reader.onload = e => {
      previewImg.src = e.target.result;

      // Switch to preview state
      dzDefault.style.display  = 'none';
      dzPreview.style.display  = 'flex';
      submitBtn.disabled       = false;

      // Inject file into hidden input if came from drag-and-drop
      if (!fileInput.files.length || fileInput.files[0] !== file) {
        try {
          const dt = new DataTransfer();
          dt.items.add(file);
          fileInput.files = dt.files;
        } catch (_) { /* Safari doesn't support DataTransfer constructor */ }
      }
    };
    reader.readAsDataURL(file);
  }

  // ── Remove image ──────────────────────────────
  removeBtn.addEventListener('click', e => {
    e.stopPropagation();
    fileInput.value  = '';
    previewImg.src   = '';
    dzDefault.style.display = '';
    dzPreview.style.display = 'none';
    submitBtn.disabled = true;
  });

  // ── Form submit ───────────────────────────────
  form.addEventListener('submit', e => {
    if (!fileInput.files.length) { e.preventDefault(); return; }

    // Show loading state
    btnText.style.display    = 'none';
    btnLoading.style.display = 'flex';
    submitBtn.disabled       = true;

    // Optional: short ripple animation on card before navigating
    const card = document.getElementById('main-card');
    if (card) {
      card.style.transform  = 'scale(.99)';
      card.style.opacity    = '.85';
      card.style.transition = 'all .3s ease';
    }
  });

  // ── Toast helper ─────────────────────────────
  function showToast(msg, type = 'info') {
    const toast = document.createElement('div');
    toast.textContent = msg;
    Object.assign(toast.style, {
      position:     'fixed',
      bottom:       '1.5rem',
      left:         '50%',
      transform:    'translateX(-50%) translateY(20px)',
      background:   type === 'error' ? 'rgba(239,68,68,.9)' : 'rgba(30,41,59,.9)',
      color:        '#fff',
      padding:      '.6rem 1.2rem',
      borderRadius: '10px',
      fontSize:     '.87rem',
      fontWeight:   '600',
      zIndex:       9999,
      opacity:      0,
      transition:   'all .3s ease',
      backdropFilter: 'blur(12px)',
      boxShadow:    '0 4px 20px rgba(0,0,0,.4)',
    });
    document.body.appendChild(toast);
    requestAnimationFrame(() => {
      toast.style.opacity   = '1';
      toast.style.transform = 'translateX(-50%) translateY(0)';
    });
    setTimeout(() => {
      toast.style.opacity   = '0';
      toast.style.transform = 'translateX(-50%) translateY(10px)';
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }
});
