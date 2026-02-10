# Comprehensive fix for health_risks.html
# This will remove all corrupted content after line 492 and replace with proper modals and JavaScript

with open('templates/health_risks.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with "Unknown" option (last good line before corruption)
corruption_line = None
for i, line in enumerate(lines):
    if '<option value="3">Unknown</option>' in line:
        corruption_line = i
        break

if corruption_line:
    # Keep everything up to and including this line
    good_content = lines[:corruption_line + 1]
    
    # Add the proper closing tags and remaining modals/JavaScript
    proper_ending = '''
                </select>
            </div>
            <button type="submit" class="btn" style="background: #ea580c; margin-top: 10px;">Get Prediction</button>
        </form>
    </div>
</div>

<!-- Modal for Skin Disease Prediction -->
<div id="skinModal"
    style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; overflow-y: auto;">
    <div
        style="background: white; max-width: 600px; margin: 50px auto; padding: 30px; border-radius: 16px; position: relative;">
        <button onclick="closeSkinModal()"
            style="position: absolute; top: 15px; right: 15px; background: none; border: none; font-size: 24px; cursor: pointer;">&times;</button>
        <h3 style="margin-bottom: 20px; color: #10b981;">Skin Disease Detection</h3>
        <p style="color: var(--text-muted); margin-bottom: 20px;">Upload three images of your face (left side, right side, and front view) for analysis:</p>

        <form id="skinForm" style="display: grid; gap: 15px;">
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Left Side Image:</label>
                <input type="file" id="left_image" name="left_image" accept="image/*" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Right Side Image:</label>
                <input type="file" id="right_image" name="right_image" accept="image/*" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Front View Image:</label>
                <input type="file" id="front_image" name="front_image" accept="image/*" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
            </div>
            <button type="submit" class="btn" style="background: #10b981; margin-top: 10px;">Analyze Images</button>
        </form>
    </div>
</div>

<script>
    // Modal Functions
    function showDiabetesForm() {
        document.getElementById('diabetesModal').style.display = 'block';
    }

    function closeDiabetesModal() {
        document.getElementById('diabetesModal').style.display = 'none';
    }

    function showHeartForm() {
        document.getElementById('heartModal').style.display = 'block';
    }

    function closeHeartModal() {
        document.getElementById('heartModal').style.display = 'none';
    }

    function showStrokeForm() {
        document.getElementById('strokeModal').style.display = 'block';
    }

    function closeStrokeModal() {
        document.getElementById('strokeModal').style.display = 'none';
    }

    function showSkinForm() {
        document.getElementById('skinModal').style.display = 'block';
    }

    function closeSkinModal() {
        document.getElementById('skinModal').style.display = 'none';
    }

    // Diabetes Prediction
    document.getElementById('diabetesForm').addEventListener('submit', async function (e) {
        e.preventDefault();
        const formData = new FormData(e.target);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/predict_diabetes', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            const resultDiv = document.getElementById('diabetes-result');

            if (result.ok) {
                const color = result.prediction === 1 ? '#dc2626' : '#16a34a';
                resultDiv.innerHTML = `
                <div style="background: ${color}15; border-left: 4px solid ${color}; padding: 15px;">
                    <h5 style="margin: 0 0 10px 0; color: ${color};">${result.message}</h5>
                    <p style="margin: 0; font-size: 14px;">${result.details}</p>
                    <div style="margin-top: 10px; font-size: 24px; font-weight: bold; color: ${color};">
                        ${result.probability}%
                    </div>
                </div>
            `;
                resultDiv.style.display = 'block';
                closeDiabetesModal();
            } else {
                alert('Error: ' + result.msg);
            }
        } catch (error) {
            alert('Prediction failed: ' + error.message);
        }
    });

    // Heart Disease Prediction
    document.getElementById('heartForm').addEventListener('submit', async function (e) {
        e.preventDefault();
        const formData = new FormData(e.target);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/predict_heart', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            const resultDiv = document.getElementById('heart-result');

            if (result.ok) {
                const color = result.prediction === 1 ? '#dc2626' : '#16a34a';
                resultDiv.innerHTML = `
                <div style="background: ${color}15; border-left: 4px solid ${color}; padding: 15px;">
                    <h5 style="margin: 0 0 10px 0; color: ${color};">${result.message}</h5>
                    <p style="margin: 0; font-size: 14px;">${result.details}</p>
                    <div style="margin-top: 10px; font-size: 24px; font-weight: bold; color: ${color};">
                        ${result.probability}%
                    </div>
                </div>
            `;
                resultDiv.style.display = 'block';
                closeHeartModal();
            } else {
                alert('Error: ' + result.msg);
            }
        } catch (error) {
            alert('Prediction failed: ' + error.message);
        }
    });

    // Stroke Prediction
    document.getElementById('strokeForm').addEventListener('submit', async function (e) {
        e.preventDefault();
        const formData = new FormData(e.target);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/predict_stroke', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            const resultDiv = document.getElementById('stroke-result');

            if (result.ok) {
                const color = result.prediction === 1 ? '#dc2626' : '#16a34a';
                resultDiv.innerHTML = `
                <div style="background: ${color}15; border-left: 4px solid ${color}; padding: 15px;">
                    <h5 style="margin: 0 0 10px 0; color: ${color};">${result.message}</h5>
                    <p style="margin: 0; font-size: 14px;">${result.details}</p>
                    <div style="margin-top: 10px; font-size: 24px; font-weight: bold; color: ${color};">
                        ${result.probability}%
                    </div>
                </div>
            `;
                resultDiv.style.display = 'block';
                closeStrokeModal();
            } else {
                alert('Error: ' + result.msg);
            }
        } catch (error) {
            alert('Prediction failed: ' + error.message);
        }
    });

    // Skin Disease Prediction
    document.getElementById('skinForm').addEventListener('submit', async function (e) {
        e.preventDefault();
        
        const leftImage = document.getElementById('left_image').files[0];
        const rightImage = document.getElementById('right_image').files[0];
        const frontImage = document.getElementById('front_image').files[0];
        
        if (!leftImage || !rightImage || !frontImage) {
            alert('Please upload all three images');
            return;
        }
        
        const formData = new FormData();
        formData.append('left_image', leftImage);
        formData.append('right_image', rightImage);
        formData.append('front_image', frontImage);
        
        try {
            const response = await fetch('/predict_skin_disease', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            const resultDiv = document.getElementById('skin-result');
            
            if (result.ok) {
                const color = result.prediction === -1 ? '#16a34a' : (result.probability > 80 ? '#dc2626' : '#ca8a04');
                resultDiv.innerHTML = `
                <div style="background: ${color}15; border-left: 4px solid ${color}; padding: 15px;">
                    <h5 style="margin: 0 0 10px 0; color: ${color};">${result.message}</h5>
                    <p style="margin: 0; font-size: 14px;">${result.details}</p>
                    <div style="margin-top: 10px; font-size: 24px; font-weight: bold; color: ${color};">
                        ${result.probability}%
                    </div>
                </div>
            `;
                resultDiv.style.display = 'block';
                closeSkinModal();
            } else {
                alert('Error: ' + result.msg);
            }
        } catch (error) {
            alert('Prediction failed: ' + error.message);
        }
    });

    // Close modals when clicking outside
    window.onclick = function (event) {
        if (event.target.id === 'diabetesModal') closeDiabetesModal();
        if (event.target.id === 'heartModal') closeHeartModal();
        if (event.target.id === 'strokeModal') closeStrokeModal();
        if (event.target.id === 'skinModal') closeSkinModal();
    }
</script>

{% endblock %}
'''
    
    # Write the fixed file
    with open('templates/health_risks.html', 'w', encoding='utf-8') as f:
        f.writelines(good_content)
        f.write(proper_ending)
    
    print("Successfully fixed health_risks.html!")
    print(f"Removed corrupted content from line {corruption_line + 1} onwards")
    print("Added proper stroke modal closing, skin disease modal, and all JavaScript")
else:
    print("Could not find corruption point")
