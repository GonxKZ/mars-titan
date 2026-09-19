"""Velas deterministas con ventana pasada, sin ejes o anotaciones futuras."""

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw


def chart_png(prices: np.ndarray, *, end_index: int, context: int = 64) -> bytes:
    if context < 2 or end_index < context - 1 or end_index >= len(prices):
        raise ValueError("Insufficient past observations for chart")
    window = np.asarray(prices[end_index - context + 1 : end_index + 1], dtype=np.float64)
    if window.shape != (context, 4) or not np.isfinite(window).all() or (window <= 0).any():
        raise ValueError("Chart needs positive finite OHLC values")
    low, high = window[:, 2].min(), window[:, 1].max()
    span = max(high - low, high * 1e-9)
    image = Image.new("RGB", (224, 224), "#fafafa")
    draw = ImageDraw.Draw(image)
    width = max(1, min(4, int(200 / context / 2)))
    for i, (opening, upper, lower, close) in enumerate(window):
        if lower > min(opening, close) or upper < max(opening, close):
            raise ValueError("Inconsistent OHLC values")
        x = round(12 + i * 200 / (context - 1))
        y = [round(212 - (price - low) * 200 / span) for price in (opening, upper, lower, close)]
        color = "#176552" if close >= opening else "#b84143"
        draw.line((x, y[1], x, y[2]), fill=color)
        draw.rectangle((x - width, min(y[0], y[3]), x + width, max(y[0], y[3])), fill=color)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()
