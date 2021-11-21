function draw(img_path){
 
    imgClo = new Image();
    imgClo.src = img_path;
    
    canvas = document.getElementById("canvas");
    ctx = canvas.getContext("2d");
    imgClo.addEventListener('load', function(){
        ctx.drawImage(imgClo , 0, 0, canvas.width, canvas.height);
    },false);
}