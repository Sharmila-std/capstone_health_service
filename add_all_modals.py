# Complete script to add all missing modals to health_risks.html

with open('templates/health_risks.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find where to insert the modals (after diabetes modal, before the script tag)
insert_line = None
for i, line in enumerate(lines):
    if '<script>' in line and i > 400:  # Make sure we're past the diabetes modal
        insert_line = i
        break

if insert_line:
    # Create the heart, stroke, and skin modals
    modals_to_add = '''
<!-- Modal for Heart Disease Prediction -->
<div id="heartModal"
    style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; overflow-y: auto;">
    <div
        style="background: white; max-width: 600px; margin: 50px auto; padding: 30px; border-radius: 16px; position: relative;">
        <button onclick="closeHeartModal()"
            style="position: absolute; top: 15px; right: 15px; background: none; border: none; font-size: 24px; cursor: pointer;">&times;</button>
        <h3 style="margin-bottom: 20px; color: #dc2626;">Heart Disease Risk Prediction</h3>
        <p style="color: var(--text-muted); margin-bottom: 20px;">Enter the following health parameters:</p>

        <form id="heartForm" style="display: grid; gap: 15px;">
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Age:</label>
                <input type="number" name="age" min="0" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Gender:</label>
                <select name="gender" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
                    <option value="0">Female</option>
                    <option value="1">Male</option>
                </select>
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Cholesterol (mg/dL):</label>
                <input type="number" name="cholesterol" min="0" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Blood Pressure (mm Hg):</label>
                <input type="number" name="blood_pressure" min="0" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Heart Rate (bpm):</label>
                <input type="number" name="heart_rate" min="0" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Smoking History:</label>
                <select name="smoking_history" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
                    <option value="0">Never</option>
                    <option value="1">Former</option>
                    <option value="2">Current</option>
                </select>
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Alcohol Intake (drinks/week):</label>
                <input type="number" name="alcohol_intake" min="0" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Exercise Hours (per week):</label>
                <input type="number" step="0.1" name="exercise_hours" min="0" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Diabetes:</label>
                <select name="diabetes" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Obesity:</label>
                <select name="obesity" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Stress Level (1-10):</label>
                <input type="number" name="stress_level" min="1" max="10" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Blood Sugar (mg/dL):</label>
                <input type="number" name="blood_sugar" min="0" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Exercise Frequency:</label>
                <select name="exercise" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
                    <option value="0">Never</option>
                    <option value="1">Rarely</option>
                    <option value="2">Sometimes</option>
                    <option value="3">Often</option>
                    <option value="4">Daily</option>
                </select>
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Family History of Heart Disease:</label>
                <select name="family_history" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>
            <button type="submit" class="btn" style="background: #dc2626; margin-top: 10px;">Get Prediction</button>
        </form>
    </div>
</div>

<!-- Modal for Stroke Prediction -->
<div id="strokeModal"
    style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; overflow-y: auto;">
    <div
        style="background: white; max-width: 600px; margin: 50px auto; padding: 30px; border-radius: 16px; position: relative;">
        <button onclick="closeStrokeModal()"
            style="position: absolute; top: 15px; right: 15px; background: none; border: none; font-size: 24px; cursor: pointer;">&times;</button>
        <h3 style="margin-bottom: 20px; color: #ea580c;">Stroke/Hypertension Risk Prediction</h3>
        <p style="color: var(--text-muted); margin-bottom: 20px;">Enter the following health parameters:</p>

        <form id="strokeForm" style="display: grid; gap: 15px;">
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Gender:</label>
                <select name="gender" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
                    <option value="0">Female</option>
                    <option value="1">Male</option>
                </select>
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Age:</label>
                <input type="number" name="age" min="0" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Hypertension:</label>
                <select name="hypertension" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Heart Disease:</label>
                <select name="heart_disease" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Ever Married:</label>
                <select name="ever_married" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Work Type:</label>
                <select name="work_type" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
                    <option value="0">Children</option>
                    <option value="1">Government Job</option>
                    <option value="2">Never Worked</option>
                    <option value="3">Private</option>
                    <option value="4">Self-employed</option>
                </select>
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Residence Type:</label>
                <select name="residence" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
                    <option value="0">Rural</option>
                    <option value="1">Urban</option>
                </select>
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Average Glucose Level (mg/dL):</label>
                <input type="number" step="0.1" name="avg_glucose" min="0" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">BMI:</label>
                <input type="number" step="0.1" name="bmi" min="0" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: 500;">Smoking History:</label>
                <select name="smoking_history" required
                    style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
                    <option value="0">Never</option>
                    <option value="1">Former</option>
                    <option value="2">Current</option>
                    <option value="3">Unknown</option>
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

'''
    
    # Insert the modals
    lines.insert(insert_line, modals_to_add)
    
    # Now add the skin disease JavaScript functions
    # Find where to add (after closeStrokeModal function)
    for i, line in enumerate(lines):
        if 'function closeStrokeModal()' in line:
            # Find the end of this function (next closing brace on its own line)
            for j in range(i+1, len(lines)):
                if lines[j].strip() == '}':
                    # Insert after this line
                    skin_js = '''
                        function showSkinForm() {
                            document.getElementById('skinModal').style.display = 'block';
                        }

                        function closeSkinModal() {
                            document.getElementById('skinModal').style.display = 'none';
                        }

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
'''
                    lines.insert(j+1, skin_js)
                    break
            break
    
    # Update window.onclick to include skinModal
    for i, line in enumerate(lines):
        if 'window.onclick = function (event) {' in line:
            # Find the closing brace
            for j in range(i+1, len(lines)):
                if lines[j].strip() == '}':
                    # Insert before the closing brace
                    lines.insert(j, "                    if (event.target.id === 'skinModal') closeSkinModal();\n")
                    break
            break
    
    # Write the updated content
    with open('templates/health_risks.html', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("Successfully added all modals and JavaScript!")
else:
    print("Could not find insertion point")
