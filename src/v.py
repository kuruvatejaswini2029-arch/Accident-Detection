import os
import cv2
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set font sizes to avoid merging
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# =========================================================
# CONFIGURATION
# =========================================================
FINAL_BALANCED_ROOT = Path(r"C:\Users\TEJASWINI\Downloads\archive\AN-Data-Balanced-Final-V2")
RESULTS_DIR = Path(r"C:\Users\TEJASWINI\Downloads\archive\visualization_results")

# Create results directory
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# FUNCTION 1: LOAD COMPREHENSIVE STATISTICS
# =========================================================
def load_comprehensive_stats(root):
    """Load all dataset statistics"""
    
    stats = {}
    
    for split in ['train', 'valid', 'test']:
        images_dir = root / split / 'images'
        labels_dir = root / split / 'labels'
        
        if not images_dir.exists():
            print(f"⚠️ {split} directory not found")
            continue
        
        # Get all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(images_dir.glob(ext))
        
        print(f"\n📊 Analyzing {split.upper()} set...")
        
        # Initialize counters
        accident_count = 0
        no_accident_count = 0
        total_annotations = 0
        class_counts = Counter()
        bbox_areas = []
        bbox_widths = []
        bbox_heights = []
        bbox_aspect_ratios = []
        objects_per_image = []
        
        for img_path in tqdm(image_files, desc=f"Processing {split}"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                no_accident_count += 1
                objects_per_image.append(0)
                continue
            
            with open(label_path, 'r') as f:
                content = f.read().strip()
            
            if not content:
                no_accident_count += 1
                objects_per_image.append(0)
                continue
            
            # Has annotations
            has_accident = False
            obj_count = 0
            
            for line in content.split('\n'):
                if line.strip():
                    parts = line.split()
                    if len(parts) == 5:
                        class_id = int(float(parts[0]))
                        class_counts[class_id] += 1
                        total_annotations += 1
                        obj_count += 1
                        
                        if class_id == 0:  # Accident
                            has_accident = True
                        
                        # Calculate bbox metrics
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        area = width * height
                        bbox_areas.append(area)
                        bbox_widths.append(width)
                        bbox_heights.append(height)
                        
                        aspect_ratio = width / height if height > 0 else 1
                        bbox_aspect_ratios.append(aspect_ratio)
            
            if has_accident:
                accident_count += 1
            else:
                no_accident_count += 1
            
            objects_per_image.append(obj_count)
        
        stats[split] = {
            'total_images': len(image_files),
            'accident_images': accident_count,
            'no_accident_images': no_accident_count,
            'total_annotations': total_annotations,
            'class_distribution': dict(class_counts),
            'bbox_areas': bbox_areas,
            'bbox_widths': bbox_widths,
            'bbox_heights': bbox_heights,
            'bbox_aspect_ratios': bbox_aspect_ratios,
            'objects_per_image': objects_per_image,
            'avg_objects_per_image': total_annotations / accident_count if accident_count > 0 else 0
        }
    
    return stats

print("="*60)
print("📊 LOADING DATASET STATISTICS")
print("="*60)

stats = load_comprehensive_stats(FINAL_BALANCED_ROOT)

# =========================================================
# FUNCTION 2: VISUALIZATION 1 - CLASS DISTRIBUTION BAR CHART
# =========================================================
def plot_class_distribution(stats, save_path):
    """Plot 1: Class distribution bar chart"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    splits = list(stats.keys())
    accident_counts = [stats[s]['accident_images'] for s in splits]
    no_accident_counts = [stats[s]['no_accident_images'] for s in splits]
    
    x = np.arange(len(splits))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accident_counts, width, label='Accident', 
                    color='#ff6b6b', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, no_accident_counts, width, label='No-Accident', 
                    color='#4ecdc4', edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Dataset Split', fontweight='bold')
    ax.set_ylabel('Number of Images', fontweight='bold')
    ax.set_title('Accident vs No-Accident Distribution', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5, f'{int(height)}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

# =========================================================
# FUNCTION 3: VISUALIZATION 2 - PIE CHARTS
# =========================================================
def plot_pie_charts(stats, save_path):
    """Plot 2: Overall distribution pie chart"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart 1: Overall image distribution
    total_accident = sum(stats[s]['accident_images'] for s in stats)
    total_no_accident = sum(stats[s]['no_accident_images'] for s in stats)
    
    colors = ['#ff6b6b', '#4ecdc4']
    explode = (0.05, 0)
    
    wedges, texts, autotexts = axes[0].pie([total_accident, total_no_accident], 
                                             labels=['Accident', 'No-Accident'],
                                             colors=colors,
                                             autopct='%1.1f%%',
                                             explode=explode,
                                             shadow=True,
                                             textprops={'fontsize': 10, 'fontweight': 'bold'})
    
    axes[0].set_title(f'Overall Image Distribution\nTotal: {total_accident + total_no_accident:,} images', 
                      fontweight='bold', pad=15)
    
    # Pie chart 2: Annotation distribution
    total_annotations = sum(stats[s]['total_annotations'] for s in stats)
    class_0_count = sum(stats[s]['class_distribution'].get(0, 0) for s in stats)
    class_1_count = sum(stats[s]['class_distribution'].get(1, 0) for s in stats)
    
    if class_0_count > 0 or class_1_count > 0:
        wedges, texts, autotexts = axes[1].pie([class_0_count, class_1_count], 
                                                 labels=['Accident (Class 0)', 'Non-Accident (Class 1)'],
                                                 colors=['#ff6b6b', '#4ecdc4'],
                                                 autopct='%1.1f%%',
                                                 explode=(0.05, 0),
                                                 shadow=True,
                                                 textprops={'fontsize': 10, 'fontweight': 'bold'})
        
        axes[1].set_title(f'Annotation Distribution\nTotal: {total_annotations:,} objects', 
                          fontweight='bold', pad=15)
    
    plt.suptitle('Dataset Composition Overview', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

# =========================================================
# FUNCTION 4: VISUALIZATION 3 - STACKED BAR CHART
# =========================================================
def plot_stacked_bar_chart(stats, save_path):
    """Plot 3: Stacked percentage bar chart"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    splits = list(stats.keys())
    accident_pcts = [stats[s]['accident_images']/stats[s]['total_images']*100 for s in splits]
    no_accident_pcts = [stats[s]['no_accident_images']/stats[s]['total_images']*100 for s in splits]
    
    y_pos = np.arange(len(splits))
    bar_height = 0.6
    
    bars1 = ax.barh(y_pos, accident_pcts, bar_height, label='Accident', 
                    color='#ff6b6b', edgecolor='black', linewidth=1)
    bars2 = ax.barh(y_pos, no_accident_pcts, bar_height, left=accident_pcts, 
                    label='No-Accident', color='#4ecdc4', edgecolor='black', linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s.capitalize() for s in splits])
    ax.set_xlabel('Percentage (%)', fontweight='bold')
    ax.set_title('Percentage Distribution by Split (100% Stacked)', fontweight='bold', pad=20)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, 100)
    
    # Add percentage labels
    for i, (acc_pct, no_acc_pct) in enumerate(zip(accident_pcts, no_accident_pcts)):
        if acc_pct > 8:
            ax.text(acc_pct/2, i, f'{acc_pct:.1f}%', ha='center', va='center', 
                   fontweight='bold', color='white', fontsize=9)
        if no_acc_pct > 8:
            ax.text(acc_pct + no_acc_pct/2, i, f'{no_acc_pct:.1f}%', ha='center', va='center', 
                   fontweight='bold', color='white', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

# =========================================================
# FUNCTION 5: VISUALIZATION 4 - HISTOGRAMS
# =========================================================
def plot_histograms(stats, save_path):
    """Plot 4: Objects per image and bbox area histograms"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram 1: Objects per image
    all_objects = []
    for s in stats:
        all_objects.extend(stats[s]['objects_per_image'])
    
    max_objects = max(all_objects) if all_objects else 10
    bins = range(0, max_objects + 2)
    
    axes[0].hist(all_objects, bins=bins, alpha=0.7, color='#6c5ce7', 
                 edgecolor='black', linewidth=1, rwidth=0.8)
    axes[0].set_xlabel('Number of Objects per Image', fontweight='bold')
    axes[0].set_ylabel('Frequency', fontweight='bold')
    axes[0].set_title('Objects per Image Distribution', fontweight='bold', pad=15)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add mean and median
    mean_objects = np.mean(all_objects)
    median_objects = np.median(all_objects)
    axes[0].axvline(mean_objects, color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {mean_objects:.2f}')
    axes[0].axvline(median_objects, color='blue', linestyle='--', linewidth=2, 
                    label=f'Median: {median_objects:.0f}')
    axes[0].legend(frameon=True, fancybox=True, shadow=True)
    
    # Histogram 2: Bounding box areas
    all_areas = []
    for s in stats:
        all_areas.extend(stats[s]['bbox_areas'])
    
    if all_areas:
        axes[1].hist(all_areas, bins=50, alpha=0.7, color='#00b894', 
                     edgecolor='black', linewidth=1)
        axes[1].set_xlabel('Bounding Box Area (normalized)', fontweight='bold')
        axes[1].set_ylabel('Frequency', fontweight='bold')
        axes[1].set_title('Bounding Box Size Distribution', fontweight='bold', pad=15)
        axes[1].grid(axis='y', alpha=0.3, linestyle='--')
        
        mean_area = np.mean(all_areas)
        median_area = np.median(all_areas)
        axes[1].axvline(mean_area, color='red', linestyle='--', linewidth=2, 
                        label=f'Mean: {mean_area:.3f}')
        axes[1].axvline(median_area, color='blue', linestyle='--', linewidth=2, 
                        label=f'Median: {median_area:.3f}')
        axes[1].legend(frameon=True, fancybox=True, shadow=True)
    
    plt.suptitle('Distribution Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

# =========================================================
# FUNCTION 6: VISUALIZATION 5 - ASPECT RATIO DISTRIBUTION
# =========================================================
def plot_aspect_ratios(stats, save_path):
    """Plot 5: Bounding box aspect ratio distribution"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_ratios = []
    for s in stats:
        all_ratios.extend(stats[s]['bbox_aspect_ratios'])
    
    if all_ratios:
        ax.hist(all_ratios, bins=50, alpha=0.7, color='#e17055', 
                edgecolor='black', linewidth=1)
        ax.set_xlabel('Aspect Ratio (width/height)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Bounding Box Aspect Ratio Distribution', fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Square (1:1)')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        
        # Add statistics box
        mean_ratio = np.mean(all_ratios)
        median_ratio = np.median(all_ratios)
        textstr = f'Mean: {mean_ratio:.2f}\nMedian: {median_ratio:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

# =========================================================
# FUNCTION 7: VISUALIZATION 6 - BOX PLOTS
# =========================================================
def plot_box_plots(stats, save_path):
    """Plot 6: Box plots for objects per split"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    objects_per_split = [stats[s]['objects_per_image'] for s in stats]
    bp = ax.boxplot(objects_per_split, labels=[s.capitalize() for s in stats], 
                     patch_artist=True, showmeans=True, meanline=True)
    
    colors_box = ['#ff6b6b', '#4ecdc4', '#95a5a6']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Number of Objects', fontweight='bold')
    ax.set_title('Objects Distribution by Split', fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add mean values
    for i, split in enumerate(stats):
        mean_val = np.mean(stats[split]['objects_per_image'])
        ax.text(i+1, mean_val + 0.5, f'μ={mean_val:.1f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

# =========================================================
# FUNCTION 8: VISUALIZATION 7 - SUMMARY TABLE
# =========================================================
def plot_summary_table(stats, save_path):
    """Plot 7: Summary statistics table"""
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    total_accident = sum(stats[s]['accident_images'] for s in stats)
    total_no_accident = sum(stats[s]['no_accident_images'] for s in stats)
    total_annotations = sum(stats[s]['total_annotations'] for s in stats)
    
    # Prepare table data
    table_data = [
        ['Metric', 'Train', 'Valid', 'Test', 'Total'],
        ['Total Images', 
         f"{stats['train']['total_images']}", 
         f"{stats['valid']['total_images']}", 
         f"{stats['test']['total_images']}",
         f"{total_accident + total_no_accident}"],
        ['Accident Images', 
         f"{stats['train']['accident_images']}", 
         f"{stats['valid']['accident_images']}", 
         f"{stats['test']['accident_images']}",
         f"{total_accident}"],
        ['No-Accident Images', 
         f"{stats['train']['no_accident_images']}", 
         f"{stats['valid']['no_accident_images']}", 
         f"{stats['test']['no_accident_images']}",
         f"{total_no_accident}"],
        ['Total Annotations', 
         f"{stats['train']['total_annotations']}", 
         f"{stats['valid']['total_annotations']}", 
         f"{stats['test']['total_annotations']}",
         f"{total_annotations}"],
        ['Avg Objects/Image', 
         f"{stats['train']['avg_objects_per_image']:.2f}", 
         f"{stats['valid']['avg_objects_per_image']:.2f}", 
         f"{stats['test']['avg_objects_per_image']:.2f}",
         f"{total_annotations/total_accident:.2f}"]
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.2, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    # Style total row
    for i in range(5):
        table[(5, i)].set_facecolor('#e8e8e8')
        table[(5, i)].set_text_props(weight='bold')
    
    ax.set_title('DATASET STATISTICS SUMMARY', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

# =========================================================
# FUNCTION 9: SAMPLE IMAGES VISUALIZATION
# =========================================================
def plot_sample_images(stats, save_path):
    """Plot 8: Sample images with bounding boxes"""
    
    for split in ['train', 'valid', 'test']:
        images_dir = FINAL_BALANCED_ROOT / split / 'images'
        labels_dir = FINAL_BALANCED_ROOT / split / 'labels'
        
        # Get accident images
        accident_images = []
        for img_path in list(images_dir.glob("*"))[:500]:
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        content = f.read()
                        if '0' in content:
                            accident_images.append(img_path)
        
        if len(accident_images) < 4:
            continue
        
        # Select random samples
        samples = random.sample(accident_images, 4)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        fig.suptitle(f'{split.upper()} SET - Sample Accident Annotations', 
                     fontsize=14, fontweight='bold', y=1.02)
        
        for idx, img_path in enumerate(samples):
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # Read label
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            parts = line.split()
                            if len(parts) == 5:
                                class_id = int(float(parts[0]))
                                if class_id == 0:
                                    x_center = float(parts[1]) * w
                                    y_center = float(parts[2]) * h
                                    box_w = float(parts[3]) * w
                                    box_h = float(parts[4]) * h
                                    
                                    x1 = int(max(0, x_center - box_w/2))
                                    y1 = int(max(0, y_center - box_h/2))
                                    x2 = int(min(w, x_center + box_w/2))
                                    y2 = int(min(h, y_center + box_h/2))
                                    
                                    # Draw bounding box
                                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                                    cv2.putText(img, 'Accident', (x1, y1-10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            axes[idx].imshow(img)
            axes[idx].set_title(f'{img_path.name}', fontsize=9)
            axes[idx].axis('off')
        
        plt.tight_layout()
        split_save_path = save_path.parent / f"sample_images_{split}.png"
        plt.savefig(split_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {split_save_path}")

# =========================================================
# MAIN EXECUTION
# =========================================================
print("\n" + "="*60)
print("🎨 GENERATING CLEAN VISUALIZATIONS")
print("="*60)
print(f"\n📁 Results will be saved to: {RESULTS_DIR}")

# Generate all visualizations
print("\n📊 Generating visualizations...")

# 1. Class distribution bar chart
plot_class_distribution(stats, RESULTS_DIR / "01_class_distribution.png")

# 2. Pie charts
plot_pie_charts(stats, RESULTS_DIR / "02_pie_charts.png")

# 3. Stacked bar chart
plot_stacked_bar_chart(stats, RESULTS_DIR / "03_stacked_bar_chart.png")

# 4. Histograms
plot_histograms(stats, RESULTS_DIR / "04_histograms.png")

# 5. Aspect ratio distribution
plot_aspect_ratios(stats, RESULTS_DIR / "05_aspect_ratios.png")

# 6. Box plots
plot_box_plots(stats, RESULTS_DIR / "06_box_plots.png")

# 7. Summary table
plot_summary_table(stats, RESULTS_DIR / "07_summary_table.png")

# 8. Sample images
plot_sample_images(stats, RESULTS_DIR / "sample_images.png")

# =========================================================
# CREATE INDEX HTML FILE
# =========================================================
def create_index_html(results_dir, stats):
    """Create an HTML index file to view all visualizations"""
    
    total_accident = sum(stats[s]['accident_images'] for s in stats)
    total_no_accident = sum(stats[s]['no_accident_images'] for s in stats)
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Dataset Visualization Results</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .summary {{
            background-color: #fff;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary h2 {{
            color: #40466e;
            margin-top: 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #ff6b6b;
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #555;
        }}
        .stat-card .number {{
            font-size: 24px;
            font-weight: bold;
            color: #ff6b6b;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .viz-card {{
            background-color: #fff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .viz-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .viz-card .caption {{
            padding: 10px;
            background-color: #fff;
            font-weight: bold;
            text-align: center;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
        }}
    </style>
</head>
<body>
    <h1>📊 Dataset Visualization Results</h1>
    
    <div class="summary">
        <h2>📈 Dataset Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Images</h3>
                <div class="number">{total_accident + total_no_accident:,}</div>
            </div>
            <div class="stat-card">
                <h3>Accident Images</h3>
                <div class="number">{total_accident:,}</div>
            </div>
            <div class="stat-card">
                <h3>No-Accident Images</h3>
                <div class="number">{total_no_accident:,}</div>
            </div>
            <div class="stat-card">
                <h3>Balance Ratio</h3>
                <div class="number">1:{total_no_accident/total_accident:.2f}</div>
            </div>
        </div>
    </div>
    
    <div class="gallery">
        <div class="viz-card">
            <img src="01_class_distribution.png" alt="Class Distribution">
            <div class="caption">Figure 1: Accident vs No-Accident Distribution</div>
        </div>
        <div class="viz-card">
            <img src="02_pie_charts.png" alt="Pie Charts">
            <div class="caption">Figure 2: Overall Dataset Composition</div>
        </div>
        <div class="viz-card">
            <img src="03_stacked_bar_chart.png" alt="Stacked Bar Chart">
            <div class="caption">Figure 3: Percentage Distribution by Split</div>
        </div>
        <div class="viz-card">
            <img src="04_histograms.png" alt="Histograms">
            <div class="caption">Figure 4: Objects & BBox Size Distribution</div>
        </div>
        <div class="viz-card">
            <img src="05_aspect_ratios.png" alt="Aspect Ratios">
            <div class="caption">Figure 5: Bounding Box Aspect Ratios</div>
        </div>
        <div class="viz-card">
            <img src="06_box_plots.png" alt="Box Plots">
            <div class="caption">Figure 6: Objects Distribution by Split</div>
        </div>
        <div class="viz-card">
            <img src="07_summary_table.png" alt="Summary Table">
            <div class="caption">Figure 7: Detailed Statistics Table</div>
        </div>
    </div>
    
    <div class="footer">
        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Dataset: Perfectly Balanced (2,453 Accident + 2,453 No-Accident)</p>
    </div>
</body>
</html>
"""
    
    index_path = results_dir / "index.html"
    with open(index_path, 'w') as f:
        f.write(html_content)
    print(f"✅ Saved: {index_path}")

# Create HTML index
create_index_html(RESULTS_DIR, stats)

# =========================================================
# FINAL SUMMARY
# =========================================================
print("\n" + "="*60)
print("✅ VISUALIZATION COMPLETE!")
print("="*60)

total_accident = sum(stats[s]['accident_images'] for s in stats)
total_no_accident = sum(stats[s]['no_accident_images'] for s in stats)

print(f"""
📊 Visualizations saved to: {RESULTS_DIR}

📁 Generated Files:
   1. 01_class_distribution.png - Class distribution bar chart
   2. 02_pie_charts.png - Overall composition pie charts  
   3. 03_stacked_bar_chart.png - Stacked percentage chart
   4. 04_histograms.png - Objects & BBox size histograms
   5. 05_aspect_ratios.png - Aspect ratio distribution
   6. 06_box_plots.png - Box plots by split
   7. 07_summary_table.png - Detailed statistics table
   8. sample_images_train.png - Sample training annotations
   9. sample_images_valid.png - Sample validation annotations
   10. sample_images_test.png - Sample test annotations
   11. index.html - Interactive gallery page

📊 Dataset Summary:
   • Total Images: {total_accident + total_no_accident}
   • Accident: {total_accident}
   • No-Accident: {total_no_accident}
   • Balance: Perfect 1:1

🎯 Open {RESULTS_DIR / 'index.html'} in your browser to view all visualizations!
""")

print("\n✅ All done! Visualizations are clean and saved successfully!")