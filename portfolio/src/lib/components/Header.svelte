<script>
	import Hamburger from './Hamburger.svelte';
	import Menu from './Menu.svelte';
	import { dataStore } from '../../stores/dataStore';

	let personal = $dataStore.PERSONAL;

	let open = false;

	const toggleSidebar = () => {
		open = !open;
	};
</script>

<header id="header">
	<nav
		class="bg-white dark:bg-gray-900 text-gray-700 dark:text-gray-200 border-gray-100 dark:border-gray-700 divide-gray-100 dark:divide-gray-700 w-full px-0 sm:px-0 font-mono py-16"
	>
		<div class="mx-auto flex flex-wrap justify-between flex-row-reverse items-center container">
			<a href="/" class="flex items-center flex-1"
				><span class="self-center whitespace-nowrap text-xl font-semibold dark:text-white">
					{personal.name}
				</span>
			</a>

			<Hamburger on:click={toggleSidebar} />
		</div>
	</nav>
	<Menu bind:open />
</header>
