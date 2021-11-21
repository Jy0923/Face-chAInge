function loadFile(input) {
    
    var file = input.files[0];
    var name = document.getElementById('fileName');
    name.textContent = file.name;

    var newImage = document.createElement("img");
    newImage.setAttribute("class", 'img');
    newImage.src = URL.createObjectURL(file);   
    newImage.style.width = "70%";
    newImage.style.height = "70%";
    newImage.style.objectFit = "contain";
 
    var button = document.getElementById("button");
    button.style.visibility = "hidden";

    var container = document.getElementById('image-show');
    container.appendChild(newImage);
};
