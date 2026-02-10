# This script will properly add the skin disease modal and JavaScript to health_risks.html

with open('templates/health_risks.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Find where to insert the skin modal (after stroke modal ends, before script tag)
stroke_modal_end = content.find('</div>\n                    </div>\n\n                    <script>')

if stroke_modal_end == -1:
    # Try alternative pattern
    stroke_modal_end = content.find('</form>\n                        </div>\n                    </div>\n\n                    <script>')

if stroke_modal_end != -1:
    # Insert position is right before the script tag
    insert_pos = content.find('<script>', stroke_modal_end)
    
    skin_modal = '''
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
    
    # Insert the modal
    new_content = content[:insert_pos] + skin_modal + content[insert_pos:]
    
    # Now add the JavaScript functions for skin disease prediction
    # Find where to add the new JS (after closeStrokeModal function)
    close_stroke_func = new_content.find('function closeStrokeModal() {')
    if close_stroke_func != -1:
        # Find the end of this function
        func_end = new_content.find('}', close_stroke_func) + 1
        # Find the next newline
        next_newline = new_content.find('\n', func_end)
        
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
        
        new_content = new_content[:next_newline] + skin_js + new_content[next_newline:]
        
        # Update the window.onclick to include skinModal
        window_onclick = new_content.find('window.onclick = function (event) {')
        if window_onclick != -1:
            # Find the closing brace of this function
            func_start = new_content.find('{', window_onclick)
            func_end = new_content.find('}', func_start)
            
            # Add skin modal check
            skin_check = "\n                            if (event.target.id === 'skinModal') closeSkinModal();"
            new_content = new_content[:func_end] + skin_check + new_content[func_end:]
    
    # Write the updated content
    with open('templates/health_risks.html', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Successfully added skin disease modal and JavaScript!")
else:
    print("Could not find insertion point")
