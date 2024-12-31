// Define variables for DOM elements
const remedyResultDiv = document.getElementById("remedyResult");

// Define navigateToNextPage function
function navigateToNextPage() {
  const nextSection = document.getElementById("predictionSection");
  if (nextSection) {
    nextSection.scrollIntoView({ behavior: "smooth" });
  } else {
    console.error("The next section element was not found.");
  }
}


//making go to top button
const goTOTopBtn = document.getElementById("go-to-top");
  console.log(goTOTopBtn);

  window.onscroll = () => {
    scrollFunction();
  };
  function scrollFunction() {
    if (
      document.body.scrollTop > 300 ||
      document.documentElement.scrollTop > 300
    ) {
      goTOTopBtn.style.display = "block";
    } else {
      goTOTopBtn.style.display = "none";
    }
  }
  goTOTopBtn.onclick = () => {
    goTOTopBtn.style.display = "none";
    window.scroll({
      top: 0,
      behavior: "smooth",
    });
  };


  document.getElementById("image").addEventListener("change", function (event) {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = function (e) {
        // Dynamically create or update the preview section
        let previewImg = document.getElementById("previewImg");
        if (!previewImg) {
          previewImg = document.createElement("img");
          previewImg.id = "previewImg";
          previewImg.style.maxWidth = "100%";
          previewImg.style.marginBottom = "20px";
          
          // Reference the form element correctly
          const form = document.getElementById("uploadForm");
          form.insertBefore(previewImg, form.querySelector("button")); // Insert above the Predict button
        }
  
        // Set the image source and display it
        previewImg.src = e.target.result;
        previewImg.style.display = "block";
      };
      reader.readAsDataURL(file); // Read the image file as a data URL
    }
  });
// Function to handle file upload and prediction
document.getElementById("uploadForm").addEventListener("submit", function (event) {
  event.preventDefault(); // Prevent form submission

  let formData = new FormData();
  let fileInput = document.getElementById("image");
  formData.append("file", fileInput.files[0]);

  fetch("/upload", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.blob())
    .then((data) => {
      let imageUrl = URL.createObjectURL(data);
      document.getElementById("gradCamResult").style.display = "block";
      document.getElementById("gradCamImage").src = imageUrl;

      // Call the prediction function
      predictDisease(fileInput.files[0]);
    })
    .catch((error) => {
      console.error("Error during upload:", error);
    });
});

// Function to predict disease
function predictDisease(imageFile) {
  let formData = new FormData();
  formData.append("image", imageFile);

  fetch("/predict", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.class) {
        predictionResult = data; // Store the prediction result globally
        document.getElementById("result").innerHTML = `
          <h3>Predicted Disease: ${data.class}</h3>
          <p>Confidence: ${data.confidence}%</p>
        `;
        document.getElementById("getRemedy").style.display = "inline-block";
      } else {
        document.getElementById("result").innerHTML = `<h3>Error: ${data.error}</h3>`;
      }
    })
    .catch((error) => {
      console.error("Error during prediction:", error);
    });
}

// Function to fetch remedies
window.getRemedy = () => {
  if (!predictionResult) {
    document.getElementById("remedyResult").innerHTML = `<p style="color: red;">Please make a prediction first.</p>`;
    return;
  }

  fetch("/remedy", {
    method: "POST",
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.remedy) {
        remedyResultDiv.style.display = "block";

        // Split the remedy list and construct a table
        const remedyList = data.remedy
          .split("<li>")
          .map((item) => item.replace(/<\/?[^>]+(>|$)/g, "").trim())
          .filter((item) => item);

        let tableHTML = `
          <h3>Remedies for ${predictionResult.class}:</h3>
          <table border="1" style="width: 100%; border-collapse: collapse;">
            <thead>
              <tr>
                <th>Remedy</th>
              </tr>
            </thead>
            <tbody>
        `;

        remedyList.forEach((remedy) => {
          tableHTML += `
            <tr>
              <td>${remedy}</td>
            </tr>
          `;
        });

        tableHTML += `
            </tbody>
          </table>
        `;

        remedyResultDiv.innerHTML = tableHTML;
      } else {
        remedyResultDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
      }
    })
    .catch((error) => {
      console.error("Error fetching remedy:", error);
      remedyResultDiv.innerHTML = `<p style="color: red;">Could not fetch remedy. Please try again.</p>`;
    });
};
