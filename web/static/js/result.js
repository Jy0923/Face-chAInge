function draw(img_path){
 
    imgClo = new Image();
    imgClo.src = img_path;
    
    ctx = canvas.getContext("2d");
    imgClo.addEventListener('load', function(){
        ctx.drawImage(imgClo , 0, 0, 500, 500);
    },false);
}