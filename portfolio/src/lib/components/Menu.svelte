<script lang="ts">
	import { fly, fade } from 'svelte/transition';
	import { Icon } from 'svelte-icons-pack';
	import { FaSolidXmark } from 'svelte-icons-pack/fa';
	import { dataStore } from '$lib/stores/dataStore';
	import { scrollToElement } from '$lib/utils/scrollUtils';
	import MenuLink from './MenuLink.svelte';
	import Divider from './Divider.svelte';

	let personal = $dataStore.PERSONAL;
	let meta = $dataStore.META;
	let navlist = $dataStore.NAVLIST;

	export let open = false;

	export let onClose = () => {
		open = false;
	};

	function handleNavClick(event: MouseEvent, href: string): void {
		event.preventDefault();
		scrollToElement(href, 'smooth', onClose);
	}
</script>

<div class="flex items-center justify-center">
	{#if open}
		<!-- Sidebar Overlay -->
		<div class="fixed inset-0 z-50 overflow-hidden">
			<!-- Use fade transition for overlay -->
			<div
				role="button"
				tabindex="0"
				aria-label="Close overlay"
				class="absolute inset-0 bg-black bg-opacity-75 transition-opacity"
				in:fade={{ duration: 300 }}
				out:fade={{ duration: 300 }}
				on:keydown={onClose}
				on:click={onClose}
			></div>

			<!-- Sidebar Content -->
			<section class="absolute inset-y-0 right-0 flex max-w-full">
				<div
					class="w-screen max-w-md"
					in:fly={{ x: 100, duration: 300 }}
					out:fly={{ x: 100, duration: 300 }}
				>
					<div class="flex h-full flex-col bg-white shadow-xl">
						<!-- Sidebar Header -->
						<div class="flex items-center justify-between px-4 py-6">
							<h2 class="text-xl font-semibold uppercase text-black">MENU</h2>
							<button on:click={onClose} class="text-gray-500 hover:text-gray-700">
								<Icon src={FaSolidXmark} className="h-7 w-7" />
							</button>
						</div>

						<!-- Sidebar Content -->
						<div class="overflow-auto px-4">
							<ul class="space-y-4">
								<Divider />
								{#each navlist as item, i}
									<MenuLink
										index={i}
										title={item.title}
										href={item.href}
										on:click={(event) => handleNavClick(event, item.href)}
									/>
									<Divider />
								{/each}
							</ul>
						</div>

						<!-- Sidebar Footer -->
						<div class="px-4 py-6">
							<p class="text-sm text-gray-500">
								&copy; {meta.creation_year}
								{personal.name}. All rights reserved.
							</p>
						</div>
					</div>
				</div>
			</section>
		</div>
	{/if}
</div>

<style>
	/* Add your custom styles here */
</style>
