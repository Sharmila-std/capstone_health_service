import re

# Read the corrupted file
with open('templates/health_risks.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the position where we need to insert the skin disease card
# It should be after the Stroke Prediction Card and before the closing </div> of the grid

# First, let's find where the corruption starts (line 168-169 where button contains modal)
# We need to fix this by properly closing the heart disease card and adding stroke + skin cards

# Find the diabetes card (this should be intact)
diabetes_start = content.find('<!-- Diabetes Prediction Card -->')
heart_start = content.find('<!-- Heart Disease Prediction Card -->')

# Extract everything before the heart disease card
before_heart = content[:heart_start]

# Now we need to rebuild from the heart disease card onwards
# Let's create the proper structure

heart_disease_card = '''            <!-- Heart Disease Prediction Card -->
            <div style="background: white; padding: 25px; border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                <h4 style="margin-bottom: 20px; color: #dc2626; text-align: center;">❤️ Heart Disease Prediction</h4>
                <button onclick="showHeartForm()" class="btn" style="width: 100%; background: #dc2626;">
                    Predict Heart Disease Risk
                </button>
                <div id="heart-result" style="margin-top: 15px; padding: 15px; border-radius: 8px; display: none;">
                </div>
            </div>

            <!-- Stroke Prediction Card -->
            <div style="background: white; padding: 25px; border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                <h4 style="margin-bottom: 20px; color: #ea580c; text-align: center;">🧠 Stroke/Hypertension Prediction
                </h4>
                <button onclick="showStrokeForm()" class="btn" style="width: 100%; background: #ea580c;">
                    Predict Stroke Risk
                </button>
                <div id="stroke-result" style="margin-top: 15px; padding: 15px; border-radius: 8px; display: none;">
                </div>
            </div>

            <!-- Skin Disease Prediction Card -->
            <div style="background: white; padding: 25px; border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                <h4 style="margin-bottom: 20px; color: #10b981; text-align: center;">🔍 Skin Disease Detection</h4>
                <button onclick="showSkinForm()" class="btn" style="width: 100%; background: #10b981;">
                    Predict Skin Disease
                </button>
                <div id="skin-result" style="margin-top: 15px; padding: 15px; border-radius: 8px; display: none;">
                </div>
            </div>

        </div>
    </div>

</div>

'''

# Find where the modals start (they should be outside the main container)
# Look for the first modal
modal_start_pattern = r'<!-- Modal for Diabetes Prediction -->'
modal_match = re.search(modal_start_pattern, content)

if modal_match:
    # Get everything from the modals onwards
    modals_and_rest = content[modal_match.start():]
    
    # Reconstruct the file
    new_content = before_heart + heart_disease_card + modals_and_rest
    
    # Write the fixed content
    with open('templates/health_risks.html', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("File fixed successfully!")
else:
    print("Could not find modal section")
