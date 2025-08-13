document.addEventListener('DOMContentLoaded', () => {
    // --- STATE MANAGEMENT ---
    let selectedMovies = new Set();
    let currentPage = 1;
    let currentSearchType = 'top'; // 'top', or 'multi'
    let currentSearchPayload = {};
    let isLoadingMore = false;
    let canLoadMore = true;
    const PAGE_SIZE = 10;

    // --- DOM ELEMENTS ---
    const searchInput = document.getElementById('searchInput');
    const selectedContainer = document.getElementById('selected-movies-container');
    const recommendBtn = document.getElementById('recommendBtn');
    const autocompleteList = document.getElementById('autocomplete-list');
    const movieGrid = document.getElementById('movieGrid');
    const resultsHeading = document.getElementById('resultsHeading');
    const loadingMoreIndicator = document.getElementById('loading-more');
    const infiniteScrollTrigger = document.getElementById('infinite-scroll-trigger');
    
    // --- DEBOUNCER for API calls ---
    const debounce = (func, wait) => {
        let timeout;
        return (...args) => {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    };

    // --- AUTOCOMPLETE LOGIC ---
    const fetchAutocomplete = async (query) => {
        if (query.length < 2) {
            autocompleteList.style.display = 'none';
            return;
        }
        try {
            const response = await fetch(`/search?q=${encodeURIComponent(query)}`);
            const suggestions = await response.json();
            displayAutocomplete(suggestions.filter(s => !selectedMovies.has(s)));
        } catch (error) {
            console.error('Error fetching autocomplete:', error);
        }
    };
    const displayAutocomplete = (suggestions) => {
        autocompleteList.innerHTML = '';
        if (suggestions.length === 0) {
            autocompleteList.style.display = 'none';
            return;
        }
        suggestions.forEach(title => {
            const item = document.createElement('div');
            item.textContent = title;
            item.addEventListener('click', () => addMovieTag(title));
            autocompleteList.appendChild(item);
        });
        autocompleteList.style.display = 'block';
    };
    searchInput.addEventListener('input', debounce(() => fetchAutocomplete(searchInput.value), 300));
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.input-wrapper')) {
            autocompleteList.style.display = 'none';
        }
    });

    // --- TAG MANAGEMENT LOGIC ---
    const addMovieTag = (title) => {
        if (selectedMovies.has(title)) return;
        selectedMovies.add(title);
        const tag = document.createElement('div');
        tag.className = 'movie-tag';
        tag.textContent = title;
        const removeBtn = document.createElement('span');
        removeBtn.textContent = '×';
        removeBtn.onclick = () => removeMovieTag(tag, title);
        tag.appendChild(removeBtn);
        selectedContainer.appendChild(tag);
        searchInput.value = '';
        autocompleteList.style.display = 'none';
    };
    const removeMovieTag = (tagElement, title) => {
        selectedMovies.delete(title);
        tagElement.remove();
    };

    // --- CORE RECOMMENDATION LOGIC ---
    const getRecommendations = async (isLoadMore = false) => {
        if (isLoadingMore) return;

        if (!isLoadMore) {
            currentPage = 1;
            canLoadMore = true;
            movieGrid.innerHTML = '';
            hideError();
            determineSearchTypeAndPayload();
            showLoading(); // Show main spinner for a new search
            hideResults();
        } else {
            showLoadingMore(); // Show small spinner at the bottom for loading more
        }

        isLoadingMore = true;

        let endpoint = '';
        let options = { method: 'POST', headers: {'Content-Type': 'application/json'} };

        if (currentSearchType === 'top') {
            endpoint = `/top-movies?page=${currentPage}&page_size=${PAGE_SIZE}`;
            options.method = 'GET';
            delete options.body;
            resultsHeading.textContent = "Top Rated Movies";
        } else {
            endpoint = `/recommend?page=${currentPage}&page_size=${PAGE_SIZE}`;
            options.body = JSON.stringify(currentSearchPayload);
            resultsHeading.textContent = "Movies You Might Like";
        }

        try {
            const response = await fetch(endpoint, options);
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to get recommendations');
            
            displayRecommendations(data.recommendations);
            canLoadMore = (typeof data.has_more === 'boolean') ? data.has_more : (data.recommendations.length >= PAGE_SIZE);
        } catch (error) {
            showError(error.message);
        } finally {
            isLoadingMore = false;
            hideLoading();
            hideLoadingMore();
        }
    };
    
    const determineSearchTypeAndPayload = () => {
        const moviesArray = Array.from(selectedMovies);
        currentSearchType = (moviesArray.length === 0) ? 'top' : 'multi';
        currentSearchPayload = (moviesArray.length > 0) ? { titles: moviesArray } : {};
    };
    
    // --- UI DISPLAY LOGIC ---
    const displayRecommendations = (recommendations) => {
        if (recommendations.length === 0 && currentPage === 1) {
            showError("No recommendations found for your selection.");
            return;
        }
        recommendations.forEach(movie => {
            const movieCard = document.createElement('div');
            movieCard.className = 'movie-card';

            // Logic for poster or placeholder
            const posterHtml = movie.poster_url
                ? `<img src="${movie.poster_url}" alt="${movie.title}" loading="lazy" class="movie-poster" onerror="this.parentElement.innerHTML='<div class=\'no-poster-placeholder\'><i class=\'fas fa-film\'></i></div>'">`
                : `<div class="no-poster-placeholder"><i class="fas fa-film"></i></div>`;
            
            // Logic for Google Search button
            const googleSearchBtnHtml = `
                <button class="google-search-btn" onclick="event.stopPropagation(); window.open('https://www.google.com/search?q=${encodeURIComponent(movie.title + ' movie')}', '_blank');" title="Search on Google">
                    <i class="fab fa-google"></i>
                </button>
            `;

            // Prepare description with smart truncation
            const description = movie.overview || 'No description available.';
            let truncatedDescription;
            
            // Truncate at different lengths based on viewport/card size
            const maxLength = 150; // Shorter to ensure it fits without scrolling
            
            if (description.length > maxLength) {
                // Find the last space before maxLength to avoid cutting words
                const lastSpace = description.lastIndexOf(' ', maxLength);
                truncatedDescription = description.substring(0, lastSpace > 0 ? lastSpace : maxLength) + '...';
            } else {
                truncatedDescription = description;
            }
            
            // Assemble the card
            movieCard.innerHTML = `
                <div class="poster-container">
                    ${posterHtml}
                    ${googleSearchBtnHtml}
                    <div class="movie-description">
                        <h4 style="margin-bottom: 0.4rem; font-size: 0.95rem; font-weight: 600;">${movie.title}</h4>
                        <p style="margin: 0; opacity: 0.9;">${truncatedDescription}</p>
                    </div>
                </div>
                <div class="movie-info">
                    <h3 class="movie-title">${movie.title}</h3>
                </div>
            `;
            movieGrid.appendChild(movieCard);
        });
        // Only show results section on first page to prevent scroll jump
        if (currentPage === 1) {
            showResults();
        }
    };

    const showLoading = () => document.getElementById('loading').classList.remove('hidden');
    const hideLoading = () => document.getElementById('loading').classList.add('hidden');
    const showLoadingMore = () => loadingMoreIndicator.classList.remove('hidden');
    const hideLoadingMore = () => loadingMoreIndicator.classList.add('hidden');
    const showResults = () => document.getElementById('results').classList.remove('hidden');
    const hideResults = () => document.getElementById('results').classList.add('hidden');
    const showError = (message) => {
        const errorEl = document.getElementById('error');
        errorEl.textContent = message;
        errorEl.classList.remove('hidden');
    };
    const hideError = () => document.getElementById('error').classList.add('hidden');

    // --- INFINITE SCROLL ---
    const observer = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting && canLoadMore && !isLoadingMore && movieGrid.children.length > 0) {
            currentPage++;
            getRecommendations(true);
        }
    }, { threshold: 0.8 });

    observer.observe(infiniteScrollTrigger);

    // --- EVENT LISTENERS ---
    recommendBtn.addEventListener('click', (e) => {
        e.preventDefault();
        getRecommendations(false);
    });
});
