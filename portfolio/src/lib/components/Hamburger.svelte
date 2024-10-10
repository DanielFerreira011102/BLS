<script>
	import { onMount } from 'svelte';
	import { fly } from 'svelte/transition';

	export let className = '';

	let isAtTop = true;

	const handleScroll = () => {
		// Check the scroll position
		isAtTop = window.scrollY === 0;
	};

	onMount(() => {
		// Attach scroll event listener
		window.addEventListener('scroll', handleScroll);

		// Clean up the listener when the component is destroyed
		return () => {
			window.removeEventListener('scroll', handleScroll);
		};
	});
</script>

<button
	on:click
	class={`fixed bg-black h-11 w-12 text-xl tracking-wide leading-11 uppercase flex items-center justify-center font-mono ${className}`}
>
	<!-- Menu text with transition -->
	{#if isAtTop}
		<span
			in:fly={{ duration: 200 }}
			out:fly={{ duration: 200 }}
			class="absolute text-white top-0 right-full pr-2 leading-11 h-full flex items-center"
		>
			Menu
		</span>
	{/if}

	<!-- Hamburger icon -->
	<span class="relative block h-0.5 w-7 bg-white">
		<span class="absolute block h-0.5 w-full bg-white -top-2.5"></span>
		<span class="absolute block h-0.5 w-full bg-white top-2.5"></span>
	</span>
</button>

<style>
	/* Optional: Add any styles you want for your button */
</style>
