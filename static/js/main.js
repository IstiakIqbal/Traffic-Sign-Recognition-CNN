document.getElementById("upload-form").addEventListener("submit", function (e) {
    e.preventDefault();
    const fileInput = document.getElementById("imageUpload");
    const file = fileInput.files[0];

    const reader = new FileReader();
    reader.onload = function (e) {
        const preview = document.getElementById("preview");
        preview.src = e.target.result;
        preview.style.display = "block";
    };
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
        method: "POST",
        body: formData,
    })
        .then((res) => res.json())
        .then((data) => {
            document.getElementById("result").innerText = "Prediction: " + data.class_name;
        })
        .catch((err) => console.error(err));
});
