$(document).ready(function() {
    // Add a link around each image with class="lightbox"
    $('.lightbox').each(function() {
        var imgSrc = $(this).attr('src');
        $(this).wrap('<a href="' + imgSrc + '" data-fancybox="gallery"></a>');
    });
});