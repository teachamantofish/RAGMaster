// docs/js/page-transition.js
document.addEventListener('DOMContentLoaded', () => {
  // Find the title on initial page load and fade it in
  const title = document.querySelector('.page-title');
  if (title) {
    title.style.opacity = 0;
    title.style.animation = 'fadeInTitle 1s ease-in-out forwards';
  }
  
  // Listen for Material's own page switch event (when using instant navigation)
  document.addEventListener('DOMContentSwitch', () => {
    // Find and fade-in the title on the new page
    const newTitle = document.querySelector('.page-title');
    if (newTitle) {
      newTitle.style.opacity = 0;
      newTitle.style.animation = 'fadeInTitle 1s ease-in-out forwards';
    }
    
    // Update timestamps after page switches
    updateLastModified();
  });
  
  // Simple last-updated timestamp
  function updateLastModified() {
    const lastModDate = new Date(document.lastModified);
    const elements = document.getElementsByClassName('last-updated');
    for (let i = 0; i < elements.length; i++) {
      elements[i].textContent = lastModDate.toLocaleDateString();
    }
  }
  
  // Run initially and after page switches
  updateLastModified();
  document.addEventListener('DOMContentSwitch', updateLastModified);
});

document.addEventListener('DOMContentLoaded', function() {
  // Get the scroll element
  const scrollButton = document.getElementById('scroll');
  
  if (scrollButton) {
    // smooth opacity transition for fade-in/out
    scrollButton.style.transition = 'opacity 1s ease-in-out';
    // Handle scroll event
    window.addEventListener('scroll', function() {
      if (window.scrollY > 100) {
        // Show the button (fade in)
        scrollButton.style.display = 'block';
        setTimeout(() => {
          scrollButton.style.opacity = '1';
        }, 10);
      } else {
        // Hide the button (fade out)
        scrollButton.style.opacity = '0';
        setTimeout(() => {
          if (window.scrollY <= 100) {
            scrollButton.style.display = 'none';
          }
        }, 1000);
      }
    });
    
    // Handle click on scroll button
    scrollButton.addEventListener('click', function(e) {
      e.preventDefault();
      
      // Smooth scroll to top
      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
      
      return false;
    });
  }
});


document.addEventListener('DOMContentLoaded', setupAreaTips);
document.addEventListener('DOMContentSwitch', setupAreaTips);

function setupAreaTips() {
  document.querySelectorAll('area[data-tip-id]').forEach(area => {
    const tipId = area.getAttribute('data-tip-id');
    const tip = document.getElementById(tipId);
    if (!tip) return;
    tip.classList.remove('visible');
    const show = () => { tip.classList.add('visible'); };
    const hide = () => { tip.classList.remove('visible'); };
    area.addEventListener('mouseenter', show);
    area.addEventListener('mouseleave', hide);
  });
}




/*
// This does not work on image maps. 
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('unused').forEach(area => {
    const dialogId = area.dataset.dialog;
    if (!dialogId) return;

    const dialog = document.getElementById(dialogId);
    if (!dialog) return;

    // Open the dialog on click
    area.addEventListener('click', (event) => {
      event.preventDefault(); // Prevent default area href behavior
      if (!dialog.open) dialog.showModal();
    });

    // Optional: Close dialog on click (or you can use a close button inside)
    dialog.addEventListener('click', () => {
      if (dialog.open) dialog.close();
    });
  });
});

*/