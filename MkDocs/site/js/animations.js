// Vanilla JS version - no jQuery dependencies
document.addEventListener('DOMContentLoaded', function() {
    //console.log('DOM content loaded - animations.js');
    
    const searchTerms = ['sample.csv', 'mockup_container.pdf', 'ResolveAlways', 'FineLineRendering', 'PerDrawingObject'];
    //console.log('Looking for terms:', searchTerms);
    
    // Focus on code spans which we know contain our content
    const codeSpans = document.querySelectorAll('code span');
    //console.log('Code spans found:', codeSpans.length);
    
    // Process each span
    codeSpans.forEach(function(span) {
        const text = span.textContent;
        
        // Check if span contains (not just equals) our search terms
        searchTerms.forEach(term => {
            if (text.includes(term)) {
                //console.log('Found keyword in span:', term, 'in text:', text);
                
                // If text is exactly our term, make it bold
                if (text === term) {
                    span.style.fontWeight = 'bold';
                    span.classList.add('highlight-keyword');
                   // console.log('Applied bold to exact match');
                } 
                // If text contains our term but has quotes around it
                else if (text === `"${term}"`) {
                    span.style.fontWeight = 'bold';
                    span.classList.add('highlight-keyword');
                   //console.log('Applied bold to quoted match');
                }
                // If it contains the term but isn't an exact match
                else {
                   // console.log('Partial match - not styling');
                }
            }
        });
    });
    
    // Create the observer
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.3
    });

    // Observe pre elements
    document.querySelectorAll('pre').forEach(function(el) {
        observer.observe(el);
    });
});
