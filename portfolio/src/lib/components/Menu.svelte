<script lang="ts">
	import { fly, fade } from 'svelte/transition';
    import { Icon } from 'svelte-icons-pack';
	import { FaSolidXmark } from 'svelte-icons-pack/fa';
	import { dataStore } from '../../stores/dataStore';
	import Divider from './Divider.svelte';

	let personal = $dataStore.PERSONAL;
	let meta = $dataStore.META;
	let navlist = $dataStore.NAVLIST;

	export let open = false;
	
	export let onClose = () => {
		open = false;
	};

	function handleNavClick(event: MouseEvent, href: string): void {
    event.preventDefault(); // Prevent the default anchor behavior
    const targetElement = document.querySelector(href);

    if (targetElement) {
        // Smooth scroll to the target section
        targetElement.scrollIntoView({ behavior: 'smooth' });

        // After scrolling, close the sidebar
        setTimeout(() => {
            onClose();
        }, 500); // Timeout to allow the scroll to complete before closing
    }
}

</script>

<div class="flex justify-center items-center">
	{#if open}
		<!-- Sidebar Overlay -->
		<div class="fixed inset-0 z-50 overflow-hidden">
			<!-- Use fade transition for overlay -->
			<div
				role="button"
				tabindex="0"
				aria-label="Close overlay"
				class="absolute inset-0 bg-gray-500 bg-opacity-75 transition-opacity"
				in:fade={{ duration: 300 }}
				out:fade={{ duration: 300 }}
				on:keydown={onClose}
				on:click={onClose}
			></div>
			<!-- Sidebar Content -->
			<section class="absolute inset-y-0 right-0 max-w-full flex">
				<div
					class="w-screen max-w-md"
					in:fly={{ x: 100, duration: 300 }}
					out:fly={{ x: 100, duration: 300 }}
				>
					<div class="h-full flex flex-col bg-white shadow-xl">
						<!-- Sidebar Header -->
						<div class="flex items-center py-6 justify-between px-4">
							<h2 class="text-xl font-semibold text-black uppercase">MENU</h2>
							<button on:click={onClose} class="text-gray-500 hover:text-gray-700">
								<Icon src={FaSolidXmark} className="h-7 w-7"/>
							</button>
						</div>
						<!-- Sidebar Content -->
						<div class="px-4 overflow-auto">
							<ul	class="space-y-4">
								{#each navlist as item, i}
									<Divider />
									<li>
										<a href={item.href} class="text-lg text-black hover:text-gray-700" on:click={(event) => handleNavClick(event, item.href)}>
											<span class="mr-6 font-mono text-gray-500">{(i + 1).toString().padStart(2, '0')}.</span>
											{item.title}
										</a>
									</li>
								{/each}
								<Divider />
							</ul>
						</div>
						<!-- Sidebar Footer -->
						<div class="py-6 px-4">
							<p class="text-gray-500 text-sm">
								&copy; {meta.creation_year} {personal.name}. All rights reserved.
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
