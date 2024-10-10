<script>
	import { dataStore } from '../../stores/dataStore';
	import Hamburger from './Hamburger.svelte';
	import Menu from './Menu.svelte';

	let open = false;

	const toggleSidebar = () => {
		open = !open;
	};

	import heroBg from '$lib/images/hero-bg.png';
</script>

<header
	id="header"
	class="relative top-0 left-0 w-full h-screen bg-cover bg-center bg-fixed before:absolute before:top-0 before:left-0 before:w-full before:h-full before:bg-black before:opacity-65"
	style="background-image: url({heroBg})"
>
	<div
		class="absolute top-0 left-0 w-full h-full opacity-40 bg-gradient-to-r from-black via-black to-transparent z-0"
	></div>

	<nav class="relative z-10">
		<a
			href="/"
			class="absolute left-28 top-12 self-center whitespace-nowrap text-xl font-semibold text-white font-mono"
			>BioLaySumm</a
		>
		<Hamburger on:click={toggleSidebar} className="right-28 top-12" />
		<Menu bind:open />
	</nav>

	<div></div>
</header>
