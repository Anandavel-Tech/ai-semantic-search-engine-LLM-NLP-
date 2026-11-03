// JavaScript Interactions for AI-Powered Semantic Search Engine

document.addEventListener("DOMContentLoaded", function () {
  // 1. Alpha slider live indicator
  const alphaSlider = document.getElementById("alphaSlider");
  const alphaValueDisplay = document.getElementById("alphaValueDisplay");
  if (alphaSlider && alphaValueDisplay) {
    alphaSlider.addEventListener("input", function () {
      const val = parseFloat(alphaSlider.value);
      alphaValueDisplay.textContent = val.toFixed(2);
      const kwPercent = Math.round(val * 100);
      const semPercent = 100 - kwPercent;
      const label = document.getElementById("alphaBreakdownLabel");
      if (label) {
        label.textContent = `(${kwPercent}% Keyword / ${semPercent}% Semantic)`;
      }
    });
  }

  // 2. Top-K slider live indicator
  const topKSlider = document.getElementById("topKSlider");
  const topKDisplay = document.getElementById("topKDisplay");
  if (topKSlider && topKDisplay) {
    topKSlider.addEventListener("input", function () {
      topKDisplay.textContent = topKSlider.value;
    });
  }

  // 3. Search query suggestion pills
  const suggestionPills = document.querySelectorAll(".search-suggestion-pill");
  const searchInput = document.getElementById("searchInput");
  const searchForm = document.getElementById("searchForm");
  suggestionPills.forEach((pill) => {
    pill.addEventListener("click", function (e) {
      e.preventDefault();
      if (searchInput) {
        searchInput.value = this.getAttribute("data-query") || this.textContent.trim();
        if (searchForm) {
          searchForm.submit();
        }
      }
    });
  });

  // 4. File drag and drop visual highlights
  const dropZone = document.getElementById("fileDropZone");
  const fileInput = document.getElementById("fileInput");
  if (dropZone && fileInput) {
    ["dragenter", "dragover"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add("border-primary", "bg-light");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove("border-primary", "bg-light");
      });
    });

    dropZone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        const countSpan = document.getElementById("selectedFileCount");
        if (countSpan) {
          countSpan.textContent = `${e.dataTransfer.files.length} file(s) selected`;
        }
      }
    });

    fileInput.addEventListener("change", () => {
      const countSpan = document.getElementById("selectedFileCount");
      if (countSpan && fileInput.files) {
        countSpan.textContent = `${fileInput.files.length} file(s) selected`;
      }
    });
  }

  // 5. Render 2D PCA Cluster Scatter Plot on HTML5 Canvas
  window.renderClusterCanvas = function (canvasId, points) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !points || points.length === 0) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Find min and max for scaling
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    points.forEach((p) => {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    });

    const rangeX = maxX - minX || 1.0;
    const rangeY = maxY - minY || 1.0;
    const padding = 50;

    // Coordinate conversion
    const scaleX = (x) => padding + ((x - minX) / rangeX) * (width - 2 * padding);
    const scaleY = (y) => height - padding - ((y - minY) / rangeY) * (height - 2 * padding);

    // Draw axes
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, scaleY(0));
    ctx.lineTo(width - padding, scaleY(0));
    ctx.moveTo(scaleX(0), padding);
    ctx.lineTo(scaleX(0), height - padding);
    ctx.stroke();

    const clusterColors = [
      "#2563eb", "#10b981", "#f59e0b", "#ef4444", 
      "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"
    ];

    // Plot data points
    points.forEach((p) => {
      const cx = scaleX(p.x);
      const cy = scaleY(p.y);
      const color = clusterColors[p.cluster % clusterColors.length];

      ctx.beginPath();
      ctx.arc(cx, cy, 7, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Label title (truncated)
      ctx.font = "10px sans-serif";
      ctx.fillStyle = "#334155";
      const shortTitle = p.title.length > 18 ? p.title.substring(0, 16) + ".." : p.title;
      ctx.fillText(shortTitle, cx + 9, cy + 3);
    });
  };
});
