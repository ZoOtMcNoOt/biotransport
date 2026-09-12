// Pointer capture supports the same component gesture with mouse, pen, or touch.
// Add buttons remain the equivalent keyboard interaction.
export function connectPaletteDrag(palette, canvas, onDrop) {
  let gesture = null;
  const inside = (event) => {
    const box = canvas.getBoundingClientRect();
    return (
      event.clientX >= box.left &&
      event.clientX <= box.right &&
      event.clientY >= box.top &&
      event.clientY <= box.bottom
    );
  };
  const finish = () => {
    if (!gesture) return;
    gesture.row.classList.remove("drag-source");
    gesture.preview?.remove();
    canvas.classList.remove("drag-over");
    gesture = null;
  };
  palette.addEventListener("pointerdown", (event) => {
    const row = event.target.closest("[data-type]");
    if (!row || event.target.closest("button") || event.button !== 0) return;
    finish();
    gesture = {
      row,
      id: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      moved: false,
    };
    row.setPointerCapture(event.pointerId);
  });
  palette.addEventListener("pointermove", (event) => {
    if (!gesture || gesture.id !== event.pointerId) return;
    if (
      !gesture.moved &&
      Math.hypot(event.clientX - gesture.x, event.clientY - gesture.y) < 6
    )
      return;
    if (!gesture.moved) {
      gesture.moved = true;
      gesture.row.classList.add("drag-source");
      gesture.preview = document.createElement("div");
      gesture.preview.className = "drag-preview";
      gesture.preview.setAttribute("aria-hidden", "true");
      gesture.preview.textContent =
        gesture.row.querySelector("strong").textContent;
      document.body.append(gesture.preview);
    }
    gesture.preview.style.left = `${event.clientX + 12}px`;
    gesture.preview.style.top = `${event.clientY + 12}px`;
    canvas.classList.toggle("drag-over", inside(event));
  });
  palette.addEventListener("pointerup", (event) => {
    if (!gesture || gesture.id !== event.pointerId) return;
    const type =
      gesture.moved && inside(event) ? gesture.row.dataset.type : null;
    finish();
    if (type) onDrop(type);
  });
  palette.addEventListener("pointercancel", finish);
  palette.addEventListener("lostpointercapture", finish);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") finish();
  });
}
