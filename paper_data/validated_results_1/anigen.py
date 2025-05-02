import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.animation import FuncAnimation
from matplotlib.font_manager import FontProperties

# ========= Customizable Parameters =========
csv_path = 'valid_results_min.csv'
output_file = 'table_animation.gif'
fps = 5
highlight_color = (0.0, 1.0, 0.0)
highlight_opacity = 0.5
cell_height = 0.1
header_color = 'lightgray'
empty_cell_color = 'white'
text_color = 'black'
column_widths = [0.75, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]   # Width of each column
font_size = 24
highlight_frames = 3
pause_frames = 1
max_duration_seconds = 120

# Truncation character limits per column (adjust as needed)
max_chars_per_column = [25, 20, 20, 20, 20, 20, 20, 20, 20, 20]
# ===========================================

# 1) Create a FontProperties object for consistent text styling
my_font = FontProperties(
    family='sans-serif',
    style='normal',
    weight='normal',
    size=font_size
)

def truncate_text(text, max_chars):
    """
    Truncate `text` to `max_chars` in length. 
    If truncation occurs, we remove from the back and add '...'.
    """
    if len(text) <= max_chars:
        return text
    # Leave space for '...'
    return text[:max_chars - 3] + '...'

# Load and limit data
data = pd.read_csv(csv_path).head(30)
nrows, ncols = data.shape

fig, ax = plt.subplots(figsize=(sum(column_widths) * 10, (nrows + 1) * 0.6))
ax.axis('off')

table = Table(ax, bbox=[0, 0, 1, 1])

# Attempt to disable any automatic font sizing (if supported by your version)
try:
    table.auto_set_font_size(False)
except AttributeError:
    pass

# Create table headers
for col, col_name in enumerate(data.columns):
    cell = table.add_cell(0, col, column_widths[col], cell_height,
                          loc='center', facecolor=header_color)
    cell_text = cell.get_text()
    # Truncate header if needed (rare)
    header_str = truncate_text(str(col_name), max_chars_per_column[col])
    cell_text.set_text(header_str)
    cell_text.set_fontproperties(my_font)
    cell_text.set_color(text_color)

# Create empty data cells
for row in range(1, nrows + 1):
    for col in range(ncols):
        cell = table.add_cell(row, col, column_widths[col], cell_height,
                              loc='center', facecolor=empty_cell_color)
        cell_text = cell.get_text()
        cell_text.set_text('')
        cell_text.set_fontproperties(my_font)
        cell_text.set_color(text_color)

ax.add_table(table)

# Compute total frames
total_cells = nrows * ncols
frames_per_cell = highlight_frames + pause_frames
calculated_frames = total_cells * frames_per_cell

# Limit total frames by max duration
max_frames = int(fps * max_duration_seconds)
total_frames = min(calculated_frames, max_frames)

# Sequence of cell fill order
cell_sequence = [(r, c) for r in range(1, nrows + 1) for c in range(ncols)]

def update(frame):
    print(f"Processing frame {frame}/{total_frames}")
    cell_idx = frame // frames_per_cell
    stage = frame % frames_per_cell

    # Reset color for all empty cells
    for r in range(1, nrows + 1):
        for c in range(ncols):
            cell = table[r, c]
            if cell.get_text().get_text() == '':
                cell.set_facecolor(empty_cell_color)

    # Perform highlight/fill if within total cells
    if cell_idx < total_cells:
        r, c = cell_sequence[cell_idx]
        cell = table[r, c]
        text_obj = cell.get_text()

        if stage < highlight_frames:
            alpha = highlight_opacity * np.sin(np.pi * (stage + 1) / highlight_frames)
            cell.set_facecolor(highlight_color + (alpha,))
        else:
            cell.set_facecolor(empty_cell_color)
            raw_text = str(data.iloc[r - 1, c])
            # Truncate text based on column limit
            truncated = truncate_text(raw_text, max_chars_per_column[c])
            text_obj.set_text(truncated)
            
            # Re-apply font properties after setting text
            text_obj.set_fontproperties(my_font)
            text_obj.set_color(text_color)

    return (table,)

anim = FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=True)
anim.save(output_file, writer='pillow', fps=fps, dpi=150)
plt.close(fig)
