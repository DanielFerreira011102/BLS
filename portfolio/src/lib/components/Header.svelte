<script>
    import { Navbar, NavBrand } from "flowbite-svelte";
    import Hamburger from "./Hamburger.svelte";
    import Menu from "./Menu.svelte";

    export let personal;
    export let navlist;

    let open = false;

    const toggleSidebar = () => {
        open = !open;
    };
</script>

<header id="header">
    <Navbar class="px-0 sm:px-0 font-mono py-16">
        <NavBrand href="/">
          <span class="self-center whitespace-nowrap text-xl font-semibold dark:text-white">{personal.name}</span>
        </NavBrand>

        <Hamburger on:click={toggleSidebar} />
    </Navbar>
    <Menu bind:open />
</header>