function scrollToElement(
	href: string,
	scrollBehavior: 'smooth' | 'auto',
	onScrollEnd?: () => void
): void {
	const targetElement = document.querySelector(href);

	if (targetElement) {
		if (
			// Check if we are at the bottom of the page
			(window.innerHeight + window.scrollY >= document.body.offsetHeight &&
				// Check if the target element is already in view
				targetElement.getBoundingClientRect().top >= 0) ||
			// Check if we are already at the target element
			targetElement.getBoundingClientRect().top === 0
		) {
			// Execute the provided scroll end callback
			if (onScrollEnd) {
				onScrollEnd();
			}

			return;
		}

		// Scroll to the target element
		targetElement.scrollIntoView({ behavior: scrollBehavior });

		// Handle scroll end event
		const handleScrollEnd = () => {
			// Execute the provided scroll end callback
			if (onScrollEnd) {
				onScrollEnd();
			}

			// Clean up the scrollend event listener
			document.removeEventListener('scrollend', handleScrollEnd);
		};

		// Add scrollend listener
		document.addEventListener('scrollend', handleScrollEnd);
	}
}

export { scrollToElement };
