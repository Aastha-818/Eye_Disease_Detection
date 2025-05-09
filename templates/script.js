/* 3. JavaScript File (script.js) */

document.addEventListener('DOMContentLoaded', function() {
    // Initial animations
    animateHeroSection();
    
    // File Upload Functionality
    const fileUpload = document.getElementById('file-upload');
    const chooseFileBtn = document.getElementById('choose-file-btn');
    const fileChosen = document.getElementById('file-chosen');
    const previewImage = document.getElementById('preview-image');
    const saveBtn = document.getElementById('save-btn');

    chooseFileBtn.addEventListener('click', function() {
        fileUpload.click();
    });

    fileUpload.addEventListener('change', function() {
        if (fileUpload.files.length > 0) {
            fileChosen.textContent = fileUpload.files[0].name;
            
            // Preview the image with animation
            const file = fileUpload.files[0];
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImage.style.opacity = 0;
                    setTimeout(() => {
                        previewImage.src = e.target.result;
                        previewImage.style.opacity = 1;
                        previewImage.style.transition = 'opacity 0.5s ease';
                    }, 300);
                };
                reader.readAsDataURL(file);
            }
        } else {
            fileChosen.textContent = 'No File Chosen';
        }
    });

    saveBtn.addEventListener('click', function() {
        if (fileUpload.files.length > 0) {
            // Add save animation
            saveBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';
            saveBtn.disabled = true;
            
            setTimeout(() => {
                saveBtn.innerHTML = '<i class="fas fa-check"></i> Saved!';
                
                setTimeout(() => {
                    saveBtn.innerHTML = 'Save';
                    saveBtn.disabled = false;
                    alert('File saved successfully!');
                }, 1500);
            }, 1500);
        } else {
            alert('Please choose a file first.');
        }
    });

    // Testimonial Slider Functionality
    const dots = document.querySelectorAll('.dot');
    const prevBtn = document.querySelector('.prev');
    const nextBtn = document.querySelector('.next');
    const testimonialContainer = document.querySelector('.testimonial-container');
    const testimonialCards = document.querySelectorAll('.testimonial-card');
    
    let currentIndex = 0;
    
    // Initialize testimonial cards with animation
    testimonialCards.forEach((card, index) => {
        setTimeout(() => {
            card.style.opacity = 1;
            card.style.transform = 'translateY(0)';
        }, 300 * index);
    });
    
    // Set active dot
    function setActiveDot(index) {
        dots.forEach(dot => dot.classList.remove('active'));
        dots[index].classList.add('active');
    }
    
    // Handle dot click
    dots.forEach((dot, index) => {
        dot.addEventListener('click', () => {
            currentIndex = index;
            setActiveDot(currentIndex);
            scrollToCard(currentIndex);
        });
    });
    
    // Handle prev/next buttons with animation
    prevBtn.addEventListener('click', () => {
        if (currentIndex > 0) {
            prevBtn.classList.add('clicked');
            setTimeout(() => {
                prevBtn.classList.remove('clicked');
            }, 300);
            
            currentIndex--;
            setActiveDot(currentIndex);
            scrollToCard(currentIndex);
        }
    });
    
    nextBtn.addEventListener('click', () => {
        if (currentIndex < dots.length - 1) {
            nextBtn.classList.add('clicked');
            setTimeout(() => {
                nextBtn.classList.remove('clicked');
            }, 300);
            
            currentIndex++;
            setActiveDot(currentIndex);
            scrollToCard(currentIndex);
        }
    });
    
    // Scroll to specific card with animation
    function scrollToCard(index) {
        const cardWidth = testimonialCards[0].offsetWidth + 20; // card width + margin
        
        // Add slide animation
        testimonialContainer.style.transition = 'all 0.5s ease';
        testimonialContainer.scrollTo({
            left: index * cardWidth,
            behavior: 'smooth'
        });
        
        // Highlight the active card
        testimonialCards.forEach((card, i) => {
            if (i === index) {
                card.style.borderColor = '#F53E2D';
                card.style.boxShadow = '0 15px 30px rgba(245, 62, 45, 0.1)';
            } else {
                card.style.borderColor = '#F53E2D';
                card.style.boxShadow = 'none';
            }
        });
    }
    
    // Scroll animations
    function animateOnScroll() {
        const elements = document.querySelectorAll('.reveal, .reveal-left, .reveal-right');
        
        elements.forEach(element => {
            const elementTop = element.getBoundingClientRect().top;
            const elementVisible = 150;
            
            if (elementTop < window.innerHeight - elementVisible) {
                element.classList.add('active');
            }
        });
    }
    
    // Hero section animations
    function animateHeroSection() {
        const heroContent = document.querySelector('.hero-content');
        const heroImage = document.querySelector('.hero-image');
        
        if (heroContent && heroImage) {
            setTimeout(() => {
                heroContent.classList.add('animated');
                heroImage.classList.add('animated');
            }, 300);
        }
    }
    
    // Add scroll animation classes to elements
    function setupScrollAnimations() {
        // Add reveal classes to sections
        document.querySelectorAll('section h2').forEach(el => {
            el.classList.add('reveal');
        });
        
        document.querySelectorAll('.features-image').forEach(el => {
            el.classList.add('reveal-left');
        });
        
        document.querySelectorAll('.features-content').forEach(el => {
            el.classList.add('reveal-right');
        });
        
        document.querySelectorAll('.test-container').forEach(el => {
            el.classList.add('reveal');
        });
        
        document.querySelectorAll('.subscribe-box').forEach(el => {
            el.classList.add('reveal');
        });
    }
    
    // Cursor animation
    function setupCursorAnimation() {
        const cursor = document.createElement('div');
        cursor.className = 'custom-cursor';
        document.body.appendChild(cursor);
        
        // Add cursor styles
        const style = document.createElement('style');
        style.innerHTML = `
            .custom-cursor {
                position: fixed;
                width: 20px;
                height: 20px;
                border: 2px solid #F53E2D;
                border-radius: 50%;
                pointer-events: none;
                transform: translate(-50%, -50%);
                z-index: 9999;
                transition: width 0.3s, height 0.3s, background-color 0.3s;
                mix-blend-mode: difference;
            }
            
            .custom-cursor.active {
                width: 50px;
                height: 50px;
                background-color: rgba(245, 62, 45, 0.2);
            }
            
            a, button, .btn, .dot, .arrow-btn {
                cursor: none;
            }
        `;
        document.head.appendChild(style);
        
        // Update cursor position
        document.addEventListener('mousemove', e => {
            cursor.style.left = e.clientX + 'px';
            cursor.style.top = e.clientY + 'px';
        });
        
        // Add active class on hover over interactive elements
        const interactiveElements = document.querySelectorAll('a, button, .btn, .dot, .arrow-btn');
        interactiveElements.forEach(el => {
            el.addEventListener('mouseenter', () => {
                cursor.classList.add('active');
            });
            
            el.addEventListener('mouseleave', () => {
                cursor.classList.remove('active');
            });
        });
    }
    
    // Parallax effect
    function setupParallax() {
        window.addEventListener('scroll', () => {
            const scrollY = window.scrollY;
            
            // Parallax for hero section
            const heroSection = document.querySelector('.hero');
            if (heroSection) {
                heroSection.style.backgroundPositionY = scrollY * 0.5 + 'px';
            }
            
            // Parallax for other elements
            document.querySelectorAll('.hero-image, .features-image, .test-image').forEach(el => {
                const speed = 0.1;
                const yPos = -(scrollY * speed);
                el.style.transform = `translateY(${yPos}px)`;
            });
        });
    }
    
    // Typing animation for headings
    function setupTypingAnimation() {
        const headings = document.querySelectorAll('h1, h2');
        
        headings.forEach(heading => {
            const text = heading.textContent;
            heading.textContent = '';
            heading.style.opacity = 1;
            
            let i = 0;
            const typeWriter = () => {
                if (i < text.length) {
                    heading.textContent += text.charAt(i);
                    i++;
                    setTimeout(typeWriter, 50);
                }
            };
            
            // Start typing when element is in view
            const observer = new IntersectionObserver(entries => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        setTimeout(typeWriter, 300);
                        observer.unobserve(heading);
                    }
                });
            });
            
            observer.observe(heading);
        });
    }
    
    // Mobile menu toggle
    const mobileBreakpoint = 768;
    
    function checkScreenSize() {
        if (window.innerWidth <= mobileBreakpoint) {
            // Add mobile menu functionality if needed
        }
    }
    
    // Initialize animations
    setupScrollAnimations();
    
    // Check if user prefers reduced motion
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    
    if (!prefersReducedMotion) {
        // Setup advanced animations only if user doesn't prefer reduced motion
        setupParallax();
        // Uncomment to enable cursor animation (optional)
        // setupCursorAnimation();
    }
    
    // Check on load and resize
    checkScreenSize();
    window.addEventListener('resize', checkScreenSize);
    
    // Add scroll event listener for animations
    window.addEventListener('scroll', animateOnScroll);
    
    // Trigger initial scroll animations
    animateOnScroll();
});