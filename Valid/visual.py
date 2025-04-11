import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects

def set_style():
    """Set a clean, academic-friendly style for plots"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
def draw_layer_block(ax, x, y, width, height, label, color='lightblue', alpha=0.7, text_color='black'):
    """Draw a layer block with a label"""
    rect = Rectangle((x, y), width, height, linewidth=1, edgecolor='black', 
                    facecolor=color, alpha=alpha)
    ax.add_patch(rect)
    
    # Add text with white outline for better readability
    text = ax.text(x + width/2, y + height/2, label, ha='center', va='center', 
                  color=text_color, fontweight='bold')
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    
    return x + width/2, y + height/2  # Return center coordinates

def connect_layers(ax, start_coords, end_coords, color='black', alpha=0.5, linestyle='-'):
    """Connect two layers with an arrow"""
    arrow = FancyArrowPatch(start_coords, end_coords, 
                            arrowstyle='->', color=color, alpha=alpha,
                            connectionstyle='arc3,rad=0.1', linewidth=1.5,
                            linestyle=linestyle)
    ax.add_patch(arrow)

def visualize_baseline_model(output_path='baseline_model_architecture.png'):
    """Create visualization for the baseline model with multiple Conv1D layers"""
    set_style()
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Set up colors
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # Define layer dimensions and positions
    input_layer_x, input_layer_y = 0.5, 4
    layer_width = 2
    layer_height = 1.5
    x_spacing = 2
    
    # Draw input layer
    input_center = draw_layer_block(ax, input_layer_x, input_layer_y, layer_width, layer_height, 
                                   'Input\n(1, 1002)', 
                                   color='lightgreen', alpha=0.7)
    
    # Draw first Conv1D layer
    conv1_center = draw_layer_block(ax, input_layer_x + layer_width + x_spacing, input_layer_y, 
                                   layer_width, layer_height, 
                                   'Conv1D\n(3, kernel=6, stride=1)\nBatchNorm1D\nReLU', 
                                   color=colors[0])
    
    # Draw second Conv1D layer
    conv2_center = draw_layer_block(ax, input_layer_x + 2*(layer_width + x_spacing), input_layer_y, 
                                   layer_width, layer_height, 
                                   'Conv1D\n(5, kernel=5, stride=1)\nBatchNorm1D\nReLU', 
                                   color=colors[1])
    
    # Draw third Conv1D layer
    conv3_center = draw_layer_block(ax, input_layer_x + 3*(layer_width + x_spacing), input_layer_y, 
                                   layer_width, layer_height, 
                                   'Conv1D\n(10, kernel=4, stride=1)\nBatchNorm1D\nReLU', 
                                   color=colors[2])
    
    # Draw fourth Conv1D layer
    conv4_center = draw_layer_block(ax, input_layer_x + 4*(layer_width + x_spacing), input_layer_y, 
                                   layer_width, layer_height, 
                                   'Conv1D\n(20, kernel=4, stride=1)\nBatchNorm1D\nReLU', 
                                   color=colors[3])
    
    # Draw fifth Conv1D layer
    conv5_center = draw_layer_block(ax, input_layer_x + 5*(layer_width + x_spacing), input_layer_y, 
                                   layer_width, layer_height, 
                                   'Conv1D\n(20, kernel=4, stride=1)\nBatchNorm1D\nReLU', 
                                   color=colors[4])
    
    # Draw Flatten layer
    flatten_center = draw_layer_block(ax, input_layer_x + 6*(layer_width + x_spacing), input_layer_y, 
                                     layer_width, layer_height, 
                                     'Flatten', 
                                     color='lightyellow')
    
    # Draw first fully connected layer
    fc1_center = draw_layer_block(ax, input_layer_x + 7*(layer_width + x_spacing), input_layer_y, 
                                 layer_width, layer_height, 
                                 'Linear(flatten_size, 10)\nReLU', 
                                 color=colors[6])
    
    # Draw output layer
    output_center = draw_layer_block(ax, input_layer_x + 8*(layer_width + x_spacing), input_layer_y, 
                                    layer_width, layer_height, 
                                    'Linear(10, 2)\nOutput', 
                                    color='salmon')
    
    # Connect the layers
    connect_layers(ax, input_center, conv1_center)
    connect_layers(ax, conv1_center, conv2_center)
    connect_layers(ax, conv2_center, conv3_center)
    connect_layers(ax, conv3_center, conv4_center)
    connect_layers(ax, conv4_center, conv5_center)
    connect_layers(ax, conv5_center, flatten_center)
    connect_layers(ax, flatten_center, fc1_center)
    connect_layers(ax, fc1_center, output_center)
    
    # Set title and labels
    ax.set_title('Baseline Semiconductor Leakage Detection Model Architecture\n(Multiple Conv1D Layers)', fontsize=16)
    ax.set_xlim(0, input_layer_x + 9*(layer_width + x_spacing))
    ax.set_ylim(2, 7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved baseline model visualization to {output_path}")
    
def visualize_improved_model(conv_filters=3, fc1_size=20, fc2_size=10, dropout1=0.3, dropout2=0.1, 
                            output_path='improved_model_architecture.png'):
    """Create visualization for the improved model with single Conv1D layer"""
    set_style()
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Set up colors
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # Define layer dimensions and positions
    input_layer_x, input_layer_y = 1, 4
    layer_width = 2.5
    layer_height = 1.5
    x_spacing = 3
    
    # Draw input layer
    input_center = draw_layer_block(ax, input_layer_x, input_layer_y, layer_width, layer_height, 
                                   'Input\n(1, 1002)', 
                                   color='lightgreen', alpha=0.7)
    
    # Draw Conv1D layer
    conv1_center = draw_layer_block(ax, input_layer_x + layer_width + x_spacing, input_layer_y, 
                                   layer_width, layer_height, 
                                   f'Conv1D\n({conv_filters}, kernel=85, stride=32)\nBatchNorm1D\nReLU', 
                                   color=colors[0])
    
    # Draw Flatten layer
    flatten_center = draw_layer_block(ax, input_layer_x + 2*(layer_width + x_spacing), input_layer_y, 
                                     layer_width, layer_height, 
                                     'Flatten', 
                                     color='lightyellow')
    
    # Draw first fully connected layer
    fc1_center = draw_layer_block(ax, input_layer_x + 3*(layer_width + x_spacing), input_layer_y, 
                                 layer_width, layer_height, 
                                 f'Linear(flatten_size, {fc1_size})\nReLU\nDropout({dropout1})', 
                                 color=colors[2])
    
    # Draw second fully connected layer
    fc2_center = draw_layer_block(ax, input_layer_x + 4*(layer_width + x_spacing), input_layer_y, 
                                 layer_width, layer_height, 
                                 f'Linear({fc1_size}, {fc2_size})\nReLU\nDropout({dropout2})', 
                                 color=colors[3])
    
    # Draw output layer
    output_center = draw_layer_block(ax, input_layer_x + 5*(layer_width + x_spacing), input_layer_y, 
                                    layer_width, layer_height, 
                                    f'Linear({fc2_size}, 2)\nOutput', 
                                    color='salmon')
    
    # Connect the layers
    connect_layers(ax, input_center, conv1_center)
    connect_layers(ax, conv1_center, flatten_center)
    connect_layers(ax, flatten_center, fc1_center)
    connect_layers(ax, fc1_center, fc2_center)
    connect_layers(ax, fc2_center, output_center)
    
    # Set title and labels
    ax.set_title('Improved Semiconductor Leakage Detection Model Architecture\n(Single Conv1D Layer)', fontsize=16)
    ax.set_xlim(0, input_layer_x + 6*(layer_width + x_spacing))
    ax.set_ylim(2, 7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add parameter counts
    # Approximate parameter count calculation
    conv_params = conv_filters * 85 * 1 + conv_filters  # Conv1D weights + bias
    bn_params = 2 * conv_filters  # BatchNorm1D (gamma and beta)
    flattened_size = conv_filters * ((1002 - 85) // 32 + 1)  # ~= 29 * conv_filters
    fc1_params = flattened_size * fc1_size + fc1_size  # fc1 weights + bias
    fc2_params = fc1_size * fc2_size + fc2_size  # fc2 weights + bias
    output_params = fc2_size * 2 + 2  # Output layer weights + bias
    total_params = conv_params + bn_params + fc1_params + fc2_params + output_params
    
    # Add textbox with parameter count
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    param_text = (f"Approx. Parameter Count:\n"
                  f"Conv1D layer: {conv_params:,}\n"
                  f"BatchNorm1D: {bn_params:,}\n"
                  f"FC1 layer: {fc1_params:,}\n"
                  f"FC2 layer: {fc2_params:,}\n"
                  f"Output layer: {output_params:,}\n"
                  f"Total: {total_params:,}")
    
    ax.text(0.02, 0.02, param_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', bbox=props)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved improved model visualization to {output_path}")

def visualize_data_processing_pipeline(output_path='data_processing_pipeline.png'):
    """Create visualization for the data processing pipeline"""
    set_style()
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define block dimensions and positions
    block_width = 2
    block_height = 1
    x_spacing = 1.5
    y_spacing = 2
    start_x, start_y = 1, 6
    
    # Define pipeline stages
    stages = [
        {'name': 'Raw Device CSV Files\n(~30,000 dev#_label.csv)', 'color': 'lightgreen'},
        {'name': 'Data Analysis\n(Explore distributions)', 'color': 'lightblue'},
        {'name': 'Train/Val/Test Split\n(Patient-wise split)', 'color': 'lightblue'},
        {'name': 'Data Preprocessing\n(Normalization)', 'color': 'lightblue'},
        {'name': 'Feature Extraction\n(Extract q values)', 'color': 'salmon'},
        {'name': 'Model Training\n(Pytorch Implementation)', 'color': 'lightyellow'},
        {'name': 'Model Evaluation\n(F-Beta, Accuracy, etc.)', 'color': 'lightyellow'}
    ]
    
    # Draw stages
    centers = []
    for i, stage in enumerate(stages):
        if i == 0:
            # First stage at top
            center = draw_layer_block(ax, start_x, start_y, block_width, block_height, 
                                     stage['name'], color=stage['color'])
        elif i < 4:
            # Stages 1-3 flow downward
            y_pos = start_y - i * (block_height + 0.5)
            center = draw_layer_block(ax, start_x, y_pos, block_width, block_height,
                                     stage['name'], color=stage['color'])
        else:
            # Stages 4-6 flow rightward
            x_pos = start_x + (i-3) * (block_width + x_spacing)
            center = draw_layer_block(ax, x_pos, start_y - 3 * (block_height + 0.5), 
                                     block_width, block_height,
                                     stage['name'], color=stage['color'])
        centers.append(center)
    
    # Connect the stages
    for i in range(len(centers) - 1):
        if i < 3:
            # Vertical connections for first stages
            connect_layers(ax, centers[i], centers[i+1], color='black', alpha=0.7, linestyle='-')
        else:
            # Horizontal connections for later stages
            connect_layers(ax, centers[i], centers[i+1], color='black', alpha=0.7, linestyle='-')
    
    # Add details and explanations
    text_blocks = [
        {'x': start_x + block_width + 0.2, 'y': start_y, 'text': '• Each CSV contains t, v, q, i columns\n• Label in filename (0: Non-leaky, 1: Leaky)'},
        {'x': start_x + block_width + 0.2, 'y': start_y - 1 * (block_height + 0.5), 'text': '• Distribution analysis\n• Visualize sample data\n• Feature importance assessment'},
        {'x': start_x + block_width + 0.2, 'y': start_y - 2 * (block_height + 0.5), 'text': '• 70% training, 15% validation, 15% test\n• Split by device ID to avoid leakage'},
        {'x': start_x + block_width + 0.2, 'y': start_y - 3 * (block_height + 0.5), 'text': '• Min-max scaling of features\n• Fit scalers only on training data'},
        {'x': start_x + 2 * (block_width + x_spacing), 'y': start_y - 3 * (block_height + 0.5) - 1.2, 'text': '• Focus on q (charge) values\n• Reshape to 1D time series\n• Ensure fixed-length inputs'},
        {'x': start_x + 3 * (block_width + x_spacing), 'y': start_y - 3 * (block_height + 0.5) - 1.2, 'text': '• Data augmentation (signal flipping, noise)\n• SWA for better generalization'},
        {'x': start_x + 4 * (block_width + x_spacing), 'y': start_y - 3 * (block_height + 0.5) - 1.2, 'text': '• Confusion matrix analysis\n• F-Beta score prioritizes recall\n• Baseline vs. improved model comparison'}
    ]
    
    for block in text_blocks:
        ax.text(block['x'], block['y'], block['text'], ha='left', va='center', 
               fontsize=9, color='black', alpha=0.8)
    
    # Set title and clean up plot
    ax.set_title('Semiconductor Leakage Detection - Data Processing Pipeline', fontsize=16)
    ax.set_xlim(0.5, start_x + 6 * (block_width + x_spacing))
    ax.set_ylim(0, start_y + 2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved data processing pipeline visualization to {output_path}")

def visualize_model_comparison():
    """Create a visual comparison of both models showing key differences"""
    set_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Model parameters for comparison
    baseline_params = ['Multiple Conv1D layers (5 conv layers)',
                       'Smaller kernels (4-6) with stride=1',
                       'Increasing filter sizes (3→5→10→20→20)',
                       'No dropout layers',
                       'Preserves input sequence length',
                       'More parameters (~240K)',
                       'Less efficient feature extraction']
    
    improved_params = ['Single Conv1D layer',
                       'Larger kernel (85) with stride=32',
                       'Fixed filter count (3)',
                       'Uses dropout for regularization',
                       'Significant dimension reduction',
                       'Fewer parameters (~1.8K)',
                       'More efficient feature extraction']
    
    # Draw baseline model summary
    ax1.text(0.5, 0.95, 'Baseline Model', ha='center', va='top', fontsize=14, fontweight='bold')
    for i, param in enumerate(baseline_params):
        ax1.text(0.05, 0.85 - i*0.1, f"• {param}", fontsize=12, va='top', ha='left')
    
    # Create simplified baseline architecture diagram
    blocks = ['Input\n(1, 1002)', 'Conv1D x5\n(Multiple small\nconvolutions)', 'Flatten', 'FC\n(10)', 'Output\n(2)']
    block_positions = np.linspace(0.2, 0.8, len(blocks))
    block_y = 0.3
    block_width = 0.1
    block_height = 0.15
    
    for i, (block, x_pos) in enumerate(zip(blocks, block_positions)):
        rect = Rectangle((x_pos - block_width/2, block_y - block_height/2), 
                        block_width, block_height, 
                        linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
        ax1.add_patch(rect)
        ax1.text(x_pos, block_y, block, ha='center', va='center', fontsize=10)
        
        # Connect blocks with arrows
        if i < len(blocks) - 1:
            arrow = FancyArrowPatch((x_pos + block_width/2, block_y), 
                                    (block_positions[i+1] - block_width/2, block_y),
                                    arrowstyle='->', color='black')
            ax1.add_patch(arrow)
    
    # Draw improved model summary
    ax2.text(0.5, 0.95, 'Improved Model', ha='center', va='top', fontsize=14, fontweight='bold')
    for i, param in enumerate(improved_params):
        ax2.text(0.05, 0.85 - i*0.1, f"• {param}", fontsize=12, va='top', ha='left')
    
    # Create simplified improved architecture diagram
    blocks = ['Input\n(1, 1002)', 'Conv1D\n(3, kernel=85,\nstride=32)', 'Flatten', 'FC1\n(20)\nDropout', 'FC2\n(10)\nDropout', 'Output\n(2)']
    block_positions = np.linspace(0.15, 0.85, len(blocks))
    
    for i, (block, x_pos) in enumerate(zip(blocks, block_positions)):
        rect = Rectangle((x_pos - block_width/2, block_y - block_height/2), 
                        block_width, block_height, 
                        linewidth=1, edgecolor='black', facecolor='salmon', alpha=0.7)
        ax2.add_patch(rect)
        ax2.text(x_pos, block_y, block, ha='center', va='center', fontsize=10)
        
        # Connect blocks with arrows
        if i < len(blocks) - 1:
            arrow = FancyArrowPatch((x_pos + block_width/2, block_y), 
                                    (block_positions[i+1] - block_width/2, block_y),
                                    arrowstyle='->', color='black')
            ax2.add_patch(arrow)
    
    # Format axes
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    fig.suptitle('Model Architecture Comparison', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    output_path = 'model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved model comparison visualization to {output_path}")

def visualize_training_process(output_path='training_process.png'):
    """Create visualization of the training process with SWA"""
    set_style()
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create illustrative curves
    epochs = np.arange(0, 51)
    
    # Generate synthetic training/validation curves
    np.random.seed(42)
    train_acc = 50 + 43 * (1 - np.exp(-epochs/15)) + np.random.normal(0, 2, len(epochs))
    train_acc = np.clip(train_acc, 0, 100)
    
    val_acc = 50 + 33 * (1 - np.exp(-epochs/20)) + np.random.normal(0, 5, len(epochs)) 
    val_acc = np.clip(val_acc, 0, 100)
    
    # Create "noisy" val accuracy for illustration
    for i in range(10, len(epochs)):
        if i % 5 == 0:
            val_acc[i] -= np.random.uniform(3, 8)
        elif i % 7 == 0:
            val_acc[i] += np.random.uniform(2, 5)
    
    # SWA averaging begins at epoch 25
    swa_start = 25
    swa_acc = np.zeros_like(val_acc)
    swa_acc[:swa_start] = np.nan
    
    # Compute simulated SWA accuracies (smoothed)
    for i in range(swa_start, len(epochs)):
        # Moving average of validation accuracies
        swa_acc[i] = np.mean(val_acc[swa_start:i+1])
    
    # Plot the curves
    ax.plot(epochs, train_acc, '-', label='Training Accuracy', color='blue', linewidth=2)
    ax.plot(epochs, val_acc, '-', label='Validation Accuracy', color='red', linewidth=2)
    ax.plot(epochs, swa_acc, '--', label='SWA Model Accuracy', color='green', linewidth=2.5)
    
    # Mark SWA start
    ax.axvline(x=swa_start, color='green', linestyle='--', alpha=0.7)
    ax.text(swa_start+0.5, 40, 'SWA Start', color='green', fontweight='bold')
    
    # Highlight key points
    best_val_idx = np.argmax(val_acc)
    best_swa_idx = np.argmax(swa_acc)
    
    ax.plot(best_val_idx, val_acc[best_val_idx], 'o', markersize=8, 
           markerfacecolor='none', markeredgecolor='red', markeredgewidth=2)
    ax.text(best_val_idx+0.5, val_acc[best_val_idx]-3, f'Best Val: {val_acc[best_val_idx]:.1f}%',
           color='red')
    
    ax.plot(best_swa_idx, swa_acc[best_swa_idx], 'o', markersize=8, 
           markerfacecolor='none', markeredgecolor='green', markeredgewidth=2)
    ax.text(best_swa_idx+0.5, swa_acc[best_swa_idx]+3, f'Best SWA: {swa_acc[best_swa_idx]:.1f}%',
           color='green')
    
    # Add illustration of collected models for SWA
    swa_epochs = range(swa_start, 50, 5)  # Collect every 5 epochs
    for epoch in swa_epochs:
        ax.plot(epoch, val_acc[epoch], 'o', markersize=6, color='green', alpha=0.7)
    
    # Add annotations
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    annotation = (
        "Stochastic Weight Averaging (SWA):\n"
        "• Start at epoch 25\n"
        "• Collect model every 5 epochs\n"
        "• Average weights of collected models\n"
        "• Results in better generalization\n"
        "• Smoother validation curve\n"
        "• Often higher final performance"
    )
    ax.text(0.02, 0.02, annotation, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', bbox=props)
    
    # Set title and labels
    ax.set_title('Training Process with Stochastic Weight Averaging (SWA)', fontsize=16)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlim(0, 50)
    ax.set_ylim(40, 90)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training process visualization to {output_path}")

# Generate all visualizations
if __name__ == "__main__":
    # Create the visualizations
    visualize_baseline_model()
    visualize_improved_model()
    visualize_data_processing_pipeline()
    visualize_model_comparison()
    visualize_training_process()
    
    print("All visualizations generated successfully!")