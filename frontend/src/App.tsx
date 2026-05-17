import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import HomePage from "./pages/HomePage";
import MockStreamPage from "./pages/MockStreamPage";
import RegistriesPage from "./pages/RegistriesPage";
import RegistryDetailPage from "./pages/RegistryDetailPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<HomePage />} />
        <Route path="registries" element={<RegistriesPage />} />
        <Route path="registries/:registryId" element={<RegistryDetailPage />} />
        <Route path="stream" element={<MockStreamPage />} />
      </Route>
    </Routes>
  );
}
