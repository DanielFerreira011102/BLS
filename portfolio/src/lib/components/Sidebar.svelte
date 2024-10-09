<script>
	import { fly, fade } from 'svelte/transition';

	export let open = false;

	const closeSidebar = () => {
		open = false;
	};
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
                on:keydown={closeSidebar}
				on:click={closeSidebar}
			></div>
			<!-- Sidebar Content -->
			<section class="absolute inset-y-0 right-0 pl-10 max-w-full flex">
				<div
					class="w-screen max-w-md"
					in:fly={{ x: 100, duration: 300 }}
					out:fly={{ x: 100, duration: 300 }}
				>
					<div class="h-full flex flex-col py-6 bg-white shadow-xl">
						<!-- Sidebar Header -->
						<div class="flex items-center justify-between px-4">
							<h2 class="text-xl font-semibold text-black">Search</h2>
							<button on:click={closeSidebar} class="text-gray-500 hover:text-gray-700">
								<span class="sr-only">Close</span>
								<svg
									class="h-6 w-6"
									xmlns="http://www.w3.org/2000/svg"
									fill="none"
									viewBox="0 0 24 24"
									stroke="currentColor"
									aria-hidden="true"
								>
									<path
										stroke-linecap="round"
										stroke-linejoin="round"
										stroke-width="2"
										d="M6 18L18 6M6 6l12 12"
									></path>
								</svg>
							</button>
						</div>
						<!-- Search Input -->
						<div class="mt-4 px-4">
							<input
								type="text"
								placeholder="Search post here"
								class="w-full p-2 border border-gray-300 rounded-md focus:border-gray-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-300 transition-colors duration-300"
							/>
						</div>
						<div class="mt-4 px-4">
							<p class="ml-2 text-gray-400">Results</p>
						</div>
						<!-- Sidebar Content -->
						<div class="mt-4 px-4 h-full overflow-auto">
							<div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
								<!-- Dynamic Card Creation -->
								{#each Array(18) as _, index}
									<div
										class="bg-gray-50 hover:bg-gray-100 p-4 cursor-pointer rounded-md border border-gray-300 transition-colors duration-300"
									>
										<h3 class="text-lg font-semibold text-black mb-2">Card {index + 1}</h3>
										<p class="text-gray-600">Content for card {index + 1}.</p>
									</div>
								{/each}
							</div>
						</div>
						<!-- Sidebar Footer -->
						<div class="mt-6 px-4">
							<button
								class="flex justify-center items-center bg-black text-white rounded-md text-sm p-2 gap-1"
							>
								<svg
									width="1rem"
									height="1rem"
									viewBox="0 0 24 24"
									fill="none"
									xmlns="http://www.w3.org/2000/svg"
								>
									<path
										fill-rule="evenodd"
										clip-rule="evenodd"
										d="M3 7C3 6.44772 3.44772 6 4 6H20C20.5523 6 21 6.44772 21 7C21 7.55228 20.5523 8 20 8H4C3.44772 8 3 7.55228 3 7ZM6 12C6 11.4477 6.44772 11 7 11H17C17.5523 11 18 11.4477 18 12C18 12.5523 17.5523 13 17 13H7C6.44772 13 6 12.5523 6 12ZM9 17C9 16.4477 9.44772 16 10 16H14C14.5523 16 15 16.4477 15 17C15 17.5523 14.5523 18 14 18H10C9.44772 18 9 17.5523 9 17Z"
										fill="currentColor"
									></path>
								</svg> Filters
							</button>
						</div>
					</div>
				</div>
			</section>
		</div>
	{/if}
	<button
		class="fixed bottom-4 right-4 bg-black text-white rounded-full p-3 shadow-lg hover:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-900 transition-colors duration-300"
		on:click={() => (open = !open)}
	>
	</button>
</div>

<style>
	/* Add your custom styles here */
</style>
