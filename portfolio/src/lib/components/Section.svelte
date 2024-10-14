<script lang="ts">
	export let id: number = 0;
	export let title: string = '';
</script>

<section
	id={title.toLowerCase().replace(' ', '-')}
	class={id % 2 === 0 ? 'bg-white' : 'bg-light-gray'}
>
	<div class="container mx-auto px-12 py-24 font-sans">
		<slot />
	</div>
</section>
