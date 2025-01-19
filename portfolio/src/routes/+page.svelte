<script>
	import { dataStore } from '$lib/stores/dataStore';
	import Activities from '$lib/components/Activities.svelte';
	import Resources from '$lib/components/Resources.svelte';
	import Divider from '$lib/components/Divider.svelte';
	import About from '$lib/components/About.svelte';
	import Section from '$lib/components/Section.svelte';
	import Timeline from '$lib/components/Timeline.svelte';

	let personal = $dataStore.PERSONAL;
	let specializations = $dataStore.SKILLS.specializations.join(', ');
</script>

<svelte:head>
	<title>{personal.name} - {personal.role}</title>
	<meta
		name="description"
		content="{personal.name} - {personal.role} specializing in {specializations}. View my projects and contact info."
	/>
</svelte:head>

<!--<Intro />-->
<Section title="resources" id={0}>
	<Resources />
</Section>

<Divider />

<Section title="activities" id={1}>
	<Activities />
</Section>


<Divider />

<Section title="timeline" id={2}>
	<Timeline />
</Section>

<Divider />

<Section title="about" id={3}>
	<About />
</Section>

<Divider />
