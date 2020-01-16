/*-----------------------------------------------------------------------------------*/
/*  BACKGROUNDS
/*-----------------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------------*/
/*  TEMPLATE FUNCTIONS
/*-----------------------------------------------------------------------------------*/
jQuery(document).ready(function($){
'use strict';
  $('.vertical-center').flexVerticalCenter({ cssAttribute: 'padding-top' });
  /* Setup ScrollSpy */
  var navHeight = jQuery('.navbar-fixed-top').height();
  jQuery('body').scrollspy({ target: '#navbar-main', offset: 200 });
  jQuery('#navbar-main [href=#]').click(function (e) {
    e.preventDefault()
  });
  jQuery('body').scrollspy('refresh')

  /* Search Effect */
  jQuery('.navbar-default, #search-wrapper').addClass('opaqued');
  if(jQuery('body').hasClass('single-project')) {
    var windowsHeight = jQuery(window).height() - 168;
    jQuery('#headerwrap').css('height', windowsHeight + 'px');
    jQuery('#headerwrap').css('min-height', windowsHeight + 'px');
  } else {
    var windowsHeight = jQuery(window).height();
    jQuery('#headerwrap').css('height', windowsHeight + 'px');
    jQuery('#headerwrap').css('min-height', windowsHeight + 'px');
  }

  /* Slide Sync to Carousel */
  jQuery('.carousel .right').click(function(x) { x.preventDefault(); jQuery('body').data('backstretch').prev(); });
  jQuery('.carousel .left').click(function(x) { x.preventDefault(); jQuery('body').data('backstretch').prev(); });
  jQuery('#basic-carousel').carousel('pause');

  /* Lightbox */
  jQuery('.launch-lb').magnificPopup({
    type: 'image',
    mainClass: 'mfp-fade',
    fixedBgPos: false,
    fixedContentPos: true,
  });

});

/*-----------------------------------------------------------------------------------*/
/*  ISOTOPE
/*-----------------------------------------------------------------------------------*/
// jQuery(document).ready(function($){
// 'use strict';
//   var $container = jQuery('#portfolio-content');
//   $container.isotope({
//     itemSelector: '.col-md-3',
//     transformsEnabled: true,
//           animationOptions: {
//       duration: 400,
//       ease: 'fade',
//       queue: false,
//     },
//     columnWidth: 223,
//   });
//   jQuery('.filter a').click(function(){
//       var selector = jQuery(this).attr('data-filter');
//       $container.isotope({ filter: selector });
//       jQuery(this).parents('ul').find('a').removeClass('active');
//       jQuery(this).addClass('active');
//       return false;
//   });
// });


/*-----------------------------------------------------------------------------------*/
/*  NICESCROLL
/*-----------------------------------------------------------------------------------*/
// jQuery(document).ready(function($){
// 'use strict';
//   /* Animate Elements */
//   jQuery('.fade-up, .fade-down, .bounce-in, .flip-in').addClass('no-display');
//   jQuery('.bounce-in').one('inview', function() {
//     jQuery(this).addClass('animated bounceIn');
//   });
//   jQuery('.flip-in').one('inview', function() {
//     jQuery(this).addClass('animated flipInY');
//   });
//   jQuery('.counter').counterUp({
//     delay: 10,
//     time: 1000
//   });
//   jQuery('.fade-up').one('inview', function() {
//     jQuery(this).addClass('animated fadeInUp');
//   });
//   jQuery('.fade-down').one('inview', function() {
//     jQuery(this).addClass('animated fadeInDown');
//   });
//
//   /* Tool Tips */
//   jQuery('#single-post-nav a').tooltip();
//
//   /* Sizing */
//   var serviceWidth = jQuery('.service-icon-wrapper').width() + 30;
//   jQuery('.service-icon-wrapper .icon').css('line-height', serviceWidth + 'px');
// });

/*-----------------------------------------------------------------------------------*/
/*  CAROUSEL
/*-----------------------------------------------------------------------------------*/
jQuery(document).ready(function($){
'use strict';
  jQuery("#portfolio-carousel").owlCarousel({
    items: 3,
    pagination: false,
    navigation: true,
    navigationText: [
      "<i class='el-icon-chevron-left icon-white'></i>",
      "<i class='el-icon-chevron-right icon-white'></i>"
    ]
  });
  jQuery("#logo-slider").owlCarousel({
      items: 5,
      pagination: true,
      navigationText: [
        "<i class='el-icon-chevron-left icon-white'></i>",
        "<i class='el-icon-chevron-right icon-white'></i>"
      ]
  });
  jQuery("#testimonials-slider").owlCarousel({
      items: 1,
      pagination: true,
      navigationText: [
        "<i class='el-icon-chevron-left icon-white'></i>",
        "<i class='el-icon-chevron-right icon-white'></i>"
      ]
  });
});

/*-----------------------------------------------------------------------------------*/
/*  SINGLE GALLERY
/*-----------------------------------------------------------------------------------*/
jQuery(window).load(function($){
'use strict';
  setTimeout(function (){
    jQuery(window).on("backstretch.after", function (e, instance, index) {
    jQuery('#basic-carousel').carousel('next');
  });
  }, 5000);
});

/*-----------------------------------------------------------------------------------*/
/*  FANCY NAV
/*-----------------------------------------------------------------------------------*/
jQuery(window).scroll(function() {
'use strict';
   jQuery(document).scroll(function() { 
        var windowsHeight = jQuery(window).height();
        var scroll_pos = jQuery(this).scrollTop();
        if(scroll_pos > 100) {
            jQuery(".navbar-fixed-top").css('background-color', 'rgba(255,255,255,1.0)');            
            jQuery('.navbar-fixed-top').removeClass('opaqued');
        } else {
            jQuery(".navbar-fixed-top").css('background-color', 'rgba(255,255,255,0.0)');
            jQuery('.navbar-fixed-top').addClass('opaqued');
        }
    });
});

/*-----------------------------------------------------------------------------------*/
/*  SEARCH BAR
/*-----------------------------------------------------------------------------------*/
// jQuery(document).ready(function($){
// 'use strict';
//   jQuery('#search-wrapper, #search-wrapper input').hide();
//
//   jQuery('a.search-trigger').click(function(){
//     jQuery('#search-wrapper').slideToggle(0, function() {
//       var check=jQuery(this).is(":hidden");
//       if(check == true) {
//           jQuery('#search-wrapper input').fadeOut(1000);
//           jQuery('.navbar-fixed-top').delay(1000).css('margin-top' , '0px');
//       } else {
//         jQuery("#search-wrapper input").focus();
//         jQuery('#search-wrapper input').fadeIn(1000);
//         jQuery('.navbar-fixed-top').delay(1000).css('margin-top' , '70px');
//       }
//     });
//   });
// });

/*-----------------------------------------------------------------------------------*/
/*  CONTACT FORM
/*-----------------------------------------------------------------------------------*/
// jQuery(document).ready(function($){
// 'use strict';
//   jQuery('#contactform').submit(function(){
//     var action = jQuery(this).attr('action');
//     jQuery("#message").slideUp(750,function() {
//     jQuery('#message').hide();
//     jQuery('#submit').attr('disabled','disabled');
//     $.post(action, {
//       name: jQuery('#name').val(),
//       email: jQuery('#email').val(),
//       website: jQuery('#website').val(),
//       comments: jQuery('#comments').val()
//     },
//       function(data){
//         document.getElementById('message').innerHTML = data;
//         jQuery('#message').slideDown('slow');
//         jQuery('#submit').removeAttr('disabled');
//         if(data.match('success') != null) jQuery('#contactform').slideUp('slow');
//         jQuery(window).trigger('resize');
//       }
//     );
//     });
//     return false;
//   });
//
// });

/*-----------------------------------------------------------------------------------*/
/*  RESPONSIVE MENU
/*-----------------------------------------------------------------------------------*/

jQuery(document).ready(function() {
'use strict';
    jQuery('.mask figure').bind('touchstart touchend', function(e) {
        e.preventDefault();
        jQuery(this).toggleClass('hovereffect');
    });
});


/*-----------------------------------------------------------------------------------*/
/*  PRELOADER
/*-----------------------------------------------------------------------------------*/
// jQuery(document).ready(function($){
// 'use strict';
//   jQuery(window).load(function(){
//     jQuery('#preloader').fadeOut('slow',function(){jQuery(this).remove();});
//   });
// });
//
// jQuery(window).load(function(){
// 'use strict';
//   var parent_height = jQuery('.mask').height() - 12;
//   jQuery('.mask figcaption a').css( 'top' , parent_height/2);
// });
//
//   $(document).ready(function(){
// 'use strict';
//
// });