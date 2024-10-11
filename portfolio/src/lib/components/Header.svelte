<script lang="ts">
	import { dataStore } from '$lib/stores/dataStore';
	import { scrollToElement } from '$lib/utils/scrollUtils';
	import { Icon } from 'svelte-icons-pack';
	import { FaBrandsGithub, FaSolidArrowDown } from 'svelte-icons-pack/fa';
	import heroBg from '$lib/images/hero-bg.png';
	import Hamburger from './Hamburger.svelte';
	import Menu from './Menu.svelte';
	import SideLink from './SideLink.svelte';

	let open = false;

	const toggleSidebar = () => {
		open = !open;
	};

	let meta = $dataStore.META;
	let navlist = $dataStore.NAVLIST.filter((item) => item.isCore);

	function handleNavClick(event: MouseEvent, href: string): void {
		event.preventDefault();
		scrollToElement(href, 'smooth');
	}
</script>

<header
	id="header"
	class="relative left-0 top-0 h-screen w-full bg-cover bg-fixed bg-center before:absolute before:left-0 before:top-0 before:h-full before:w-full before:bg-black before:opacity-65"
	style="background-image: url({heroBg})"
>
	<!-- Overlay -->
	<div
		class="absolute left-0 top-0 z-0 h-full w-full bg-gradient-to-r from-black via-black to-transparent opacity-40"
	></div>

	<!-- Header -->
	<div class="absolute left-0 top-0 z-[4] h-36 w-full">
		<!-- Logo -->
		<a
			href="/"
			class="absolute left-8 top-6 h-16 self-center whitespace-nowrap font-mono text-xl font-semibold leading-16 text-white sm:left-14 lg:left-28 lg:top-12"
			>{meta.title}</a
		>

		<!-- Hamburger & Menu -->
		<Hamburger
			on:click={toggleSidebar}
			className="right-8 top-6 sm:right-14 lg:right-28 lg:top-12 h-16"
		/>
		<Menu bind:open />
	</div>

	<!-- Home Content -->
	<div class="absolute left-0 top-0 z-[1] h-full w-full">
		<div
			class="container relative mx-auto flex h-full flex-col items-start justify-center px-8 text-left sm:px-12 lg:px-24"
		>
			<h1
				class="pb-4 font-serif text-4xl font-semibold !leading-tight text-white xs:pb-1 xs:text-6xl md:text-7xl lg:text-8xl"
			>
				{meta.title}
			</h1>
			<p
				class="relative pl-0 pt-4 font-sans text-xl font-light !leading-snug text-white text-opacity-80 before:absolute before:left-0 before:top-0 before:h-0.5 before:w-20 before:bg-white xs:text-2xl sm:pl-28 sm:pt-0 sm:before:left-1.5 sm:before:top-6 md:text-3xl lg:text-4xl"
			>
				{@html meta.description}
			</p>
		</div>
	</div>

	<!-- Side Links -->
	<ul class="absolute right-0 top-1/2 z-[3] hidden -translate-y-1/2 transform font-mono xl:block">
		{#each navlist as item, i}
			<SideLink
				className="last:border-b"
				href={item.href}
				title={item.title}
				subtitle={item.subtitle}
				on:click={(event) => handleNavClick(event, item.href)}
			/>
		{/each}
	</ul>

	<!-- Footer -->
	<div class="absolute bottom-0 left-0 z-[2] h-36 w-full font-mono">
		<!-- GitHub Follow -->
		<div
			class="absolute bottom-6 left-8 h-16 uppercase leading-16 text-white sm:left-14 lg:bottom-12 lg:left-28"
		>
			<span
				class="relative float-left hidden pr-14 after:absolute after:right-0 after:top-1/2 after:h-0.5 after:w-10 after:bg-white md:inline-block"
			>
				Follow Me
			</span>
			<a href={meta.src} class="ml-4 inline-block border-white hover:border-b-2">
				<Icon src={FaBrandsGithub} size="24" className="inline-block" />
				<span>GitHub</span>
			</a>
		</div>
	</div>
</header>
