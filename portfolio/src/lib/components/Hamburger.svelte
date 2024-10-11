<script lang="ts">
	import { onMount } from 'svelte';
	import { fly } from 'svelte/transition';

	export let className = '';

	let isAtTop = true;

	const handleScroll = () => {
		isAtTop = window.scrollY === 0;
	};

	onMount(() => {
		window.addEventListener('scroll', handleScroll);

		return () => {
			window.removeEventListener('scroll', handleScroll);
		};
	});
</script>

<div class={`${className} fixed flex items-center`}>
	<button
		on:click
		class="flex h-11 w-12 items-center justify-center bg-black font-mono text-xl uppercase leading-11 tracking-widest"
	>
		<!-- Menu text with transition -->
		{#if isAtTop}
			<span
				in:fly={{ duration: 200 }}
				out:fly={{ duration: 200 }}
				class="absolute right-full top-0 hidden h-full items-center pr-2 text-white md:flex"
			>
				Menu
			</span>
		{/if}

		<!-- Hamburger icon -->
		<span class="relative block h-0.5 w-7 bg-white">
			<span class="absolute -top-2.5 block h-0.5 w-full bg-white"></span>
			<span class="absolute top-2.5 block h-0.5 w-full bg-white"></span>
		</span>
	</button>
</div>
