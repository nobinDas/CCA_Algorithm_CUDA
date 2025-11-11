# Performance Test Script for Different Image Sizes

echo "=== Connected Component Analysis - Performance Scaling Test ==="
echo ""

# Test different image sizes
sizes=("128" "256" "512" "1024")

for size in "${sizes[@]}"; do
    echo "Testing ${size}x${size} image..."
    
    # Update the image size in main.cu
    sed -i "s/#define IMG_WIDTH.*/#define IMG_WIDTH $size/" main.cu
    sed -i "s/#define IMG_HEIGHT.*/#define IMG_HEIGHT $size/" main.cu
    
    # Compile and run
    make > /dev/null 2>&1
    echo "--- Results for ${size}x${size} ---"
    ./connected_components | grep -E "(Time:|Components found:|Speedup vs CPU)"
    echo ""
done

echo "Performance scaling test complete!"